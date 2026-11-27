"""Validate rank replacement with Mooncake's NCCL GIN backend.

Example: On two nodes with eight GPUs each, ranks 4–7 and 12–15 exit after
pre_failure_round. The eight survivors create a 2×4 Rail GIN buffer and run
failure_round. Once every survivor finishes, the supervisors restart the
failed ranks in their original slots. Survivors rebuild their full-size GIN
buffer while replacements create theirs in the same TCPStore namespace.
All 16 ranks run after_recovery_round.

Every round verifies cross-node dispatch/combine. The configured failure set
must leave four GPUs on each node, a supported reduced Rail GIN topology.
On Slurm, launch one supervisor per node.
"""

import argparse
from datetime import timedelta
from enum import Enum
import os
import subprocess
import time


CRASH_EXIT_CODE = 17
DEFAULT_FAILED_RANKS = "4,5,6,7,12,13,14,15"
SURVIVORS_PER_NODE = 4
TOKENS = 8
MAX_TOKENS = 128
HIDDEN = 4096
TOPK = 8
EXPERTS_PER_RANK = 16
NUM_SMS = 24


class Round(Enum):
    PRE_FAILURE = ("pre_failure_round", -1)
    FAILURE = ("failure_round", 0)
    AFTER_RECOVERY = ("after_recovery_round", 1)

    def __init__(self, label, payload_band):
        self.label = label
        self.payload_band = payload_band


class RoundValidation:
    def __init__(self, rank, members, phase, config, *, device="cuda"):
        import torch

        self.torch = torch
        self.rank = rank
        self.label = f"{phase.label} rank {rank}"
        self.ranks_per_node = config.ranks_per_node
        self.payload_offset = phase.payload_band * config.world_size * TOKENS
        # Pair matching communicator indices across the two nodes.
        stride = len(members) // config.nodes
        self.peer_index = (members.index(rank) + stride) % len(members)
        self.peer_rank = members[self.peer_index]
        self.tokens = torch.arange(TOKENS, device=device, dtype=torch.long)
        self.lanes = torch.arange(TOPK, device=device, dtype=torch.long)
        self.column_signs = 1 - 2 * (torch.arange(HIDDEN, device=device) % 2)

    def _payload(self, source_rank, tokens):
        values = tokens + source_rank * TOKENS + self.payload_offset
        return (values[:, None] * self.column_signs).to(self.torch.bfloat16)

    def _routing_metadata(self, source_rank, tokens):
        positions = tokens[:, None] + self.lanes + source_rank
        experts = positions % EXPERTS_PER_RANK
        # Give each lane a distinct weight while keeping each row's sum at 1.
        weight_order = (positions + source_rank // self.ranks_per_node) % TOPK
        weights = (2 * weight_order + 1).to(self.torch.float32) / (TOPK * TOPK)
        return experts, weights

    def dispatch_inputs(self):
        experts, weights = self._routing_metadata(self.rank, self.tokens)
        return (
            self._payload(self.rank, self.tokens),
            self.peer_index * EXPERTS_PER_RANK + experts,
            weights,
        )

    def _assert_close(self, actual, expected, what):
        self.torch.testing.assert_close(
            actual,
            expected,
            rtol=0,
            atol=0,
            msg=lambda details: f"{self.label}: {what}\n{details}",
        )

    def prepare_combine(self, recv, recv_indices, recv_weights, handle):
        """Check dispatch and return the expert outputs and weights for combine."""
        actual = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
        if actual != TOKENS:
            raise AssertionError(
                f"{self.label}: received {actual} tokens, expected {TOKENS}"
            )
        src = handle.recv_src_metadata[:actual, 0].long()
        self._assert_close(
            src.sort().values,
            self.peer_index * MAX_TOKENS + self.tokens,
            "dispatch source metadata",
        )
        source_tokens = src % MAX_TOKENS
        self._assert_close(
            recv[:actual],
            self._payload(self.peer_rank, source_tokens),
            "dispatch payload",
        )
        expected_indices, expected_weights = self._routing_metadata(
            self.peer_rank, source_tokens
        )
        self._assert_close(
            recv_indices[:actual], expected_indices, "dispatch expert indices"
        )
        if recv_weights is None:
            raise AssertionError(f"{self.label}: dispatch returned no routing weights")
        self._assert_close(
            recv_weights[:actual], expected_weights, "dispatch routing weights"
        )
        # Negation stands in for expert compute and makes a stale input detectable.
        return recv[:actual].neg().contiguous(), recv_weights[:actual].contiguous()

    def check_combine(self, combined, original):
        self._assert_close(combined, original.neg(), "combine output")


def open_store(config, *, server=False):
    import torch.distributed as dist

    return dist.TCPStore(
        config.master_addr,
        config.port,
        None,
        server,
        timeout=timedelta(seconds=config.timeout_s),
        wait_for_workers=False,
    )


class FailoverWorker:
    """Own one logical rank, its Store client, and its ElasticBuffer."""

    def __init__(self, rank, is_replacement, config):
        import torch

        self.torch = torch
        self.rank = rank
        self.is_replacement = is_replacement
        self.config = config
        self.all_ranks = tuple(range(config.world_size))
        self.timeout = timedelta(seconds=config.timeout_s)
        self.buffer = None

        torch.cuda.set_device(rank % config.ranks_per_node)
        os.environ["MOONCAKE_EP_NUM_LOCAL_RANKS"] = str(config.ranks_per_node)
        self.store = open_store(config)

    def _make_buffer(self, namespace, members):
        from mooncake.mooncake_elastic_buffer import ElasticBuffer

        # Python transport selection needs the current group's ranks per node.
        os.environ["MOONCAKE_EP_NUM_LOCAL_RANKS"] = str(
            len(members) // self.config.nodes
        )
        try:
            return ElasticBuffer.create_elastic(
                members.index(self.rank),
                len(members),
                self.store,
                store_namespace=namespace,
                store_timeout_secs=self.config.timeout_s,
                num_max_tokens_per_rank=MAX_TOKENS,
                hidden=HIDDEN,
                num_topk=TOPK,
                use_fp8_dispatch=False,
                allow_hybrid_mode=True,
                explicitly_destroy=True,
                num_gpu_timeout_secs=30,
                transport="nccl",
            )
        finally:
            os.environ["MOONCAKE_EP_NUM_LOCAL_RANKS"] = str(self.config.ranks_per_node)

    def _prepare_buffer(self, phase, namespace, members):
        # Exclude process startup and waiting for ranks to reach this phase.
        ready = f"buffer_ready/{phase.label}"
        self.store.set(f"{ready}/{self.rank}", "1")
        self.store.wait([f"{ready}/{rank}" for rank in members], self.timeout)

        started_ns = time.perf_counter_ns()
        if phase is Round.AFTER_RECOVERY and not self.is_replacement:
            self._recover_survivor(namespace)
            buffer = self.buffer
        else:
            buffer = self._make_buffer(namespace, members)

        elapsed_ns = time.perf_counter_ns() - started_ns
        self.store.set(f"buffer_setup_ns/{phase.label}/{self.rank}", str(elapsed_ns))
        return buffer

    def _run_round(self, buffer, members, phase):
        torch = self.torch
        world_size = len(members)
        ranks_per_node = world_size // self.config.nodes

        topology = buffer.get_logical_domain_size()
        expected_topology = (self.config.nodes, ranks_per_node)
        if topology != expected_topology:
            raise AssertionError(
                f"rank {self.rank}: Rail GIN requires topology {expected_topology}, got {topology}"
            )

        check = RoundValidation(self.rank, members, phase, self.config)
        x, topk, weights = check.dispatch_inputs()

        recv, recv_indices, recv_weights, handle, _ = buffer.dispatch(
            x,
            topk_idx=topk,
            topk_weights=weights,
            num_experts=world_size * EXPERTS_PER_RANK,
            num_max_tokens_per_rank=MAX_TOKENS,
            expert_alignment=1,
            do_cpu_sync=True,
            num_sms=NUM_SMS,
            async_with_compute_stream=False,
        )
        torch.cuda.synchronize()
        expert_output, combine_weights = check.prepare_combine(
            recv, recv_indices, recv_weights, handle
        )

        combined, _, _ = buffer.combine(
            expert_output,
            handle,
            topk_weights=combine_weights,
            num_sms=NUM_SMS,
            async_with_compute_stream=False,
        )
        torch.cuda.synchronize()
        check.check_combine(combined, x)
        return x, handle

    def _report_pass(self, phase, members):
        if self.rank == members[0]:
            print(
                f"PASS {phase.label} data_path=nccl_rail_gin members={list(members)}",
                flush=True,
            )

    def _check_stale_handle(self, x, handle):
        try:
            self.buffer.dispatch(
                x,
                handle=handle,
                num_experts=self.config.world_size * EXPERTS_PER_RANK,
                num_max_tokens_per_rank=MAX_TOKENS,
                num_sms=NUM_SMS,
            )
        except RuntimeError as error:
            if "obsolete NCCL ElasticBuffer generation" not in str(error):
                raise
        else:
            raise AssertionError(
                "surviving buffer accepted a handle from the failed generation"
            )

    def _run_pre_failure_round(self):
        phase = Round.PRE_FAILURE
        self.buffer = self._prepare_buffer(phase, "failover/initial", self.all_ranks)
        stale_round = self._run_round(self.buffer, self.all_ranks, phase)
        # The round synchronizes the GPU. Fence all ranks before any exits.
        self.store.set(f"{phase.label}/{self.rank}", "1")
        self.store.wait(
            [f"{phase.label}/{rank}" for rank in self.all_ranks], self.timeout
        )
        self._report_pass(phase, self.all_ranks)
        if self.rank in self.config.failed_ranks:
            print(f"rank {self.rank} pid={os.getpid()}: crashing", flush=True)
            os._exit(CRASH_EXIT_CODE)
        return stale_round

    def _run_failure_round(self):
        phase = Round.FAILURE
        survivors = self.config.survivors
        self.store.wait(
            [f"exited/{rank}" for rank in self.config.failed_ranks], self.timeout
        )

        # Retain the original buffer for reconfiguration after this round.
        buffer = self._prepare_buffer(phase, "failover/survivors", survivors)
        print(f"make buffer for survivors, rank id: {self.rank}", flush=True)
        self._run_round(buffer, survivors, phase)
        buffer.destroy()

        self._report_pass(phase, survivors)
        self.store.set(f"{phase.label}/{self.rank}", "1")

    def _recover_survivor(self, namespace):
        from mooncake.mooncake_ep_tcpstore import TCPStoreCollectives

        class RecoveryCollectives(TCPStoreCollectives):
            def barrier(self, operation="barrier"):
                # create_elastic() calls the same recovery round "initialize".
                if operation == "nccl-update-complete":
                    operation = "initialize"
                super().barrier(operation)

        # update_ep_member() reuses the buffer's Store state. Start at round 0
        # so the survivor and replacement exchange metadata in the same order.
        self.buffer._store_namespace = namespace
        self.buffer._store_collectives = RecoveryCollectives(
            self.store,
            self.rank,
            self.config.world_size,
            namespace,
            self.config.timeout_s,
        )
        self.buffer.update_ep_member()

    def run(self):
        if not self.is_replacement:
            stale_round = self._run_pre_failure_round()
            self._run_failure_round()

        phase = Round.AFTER_RECOVERY
        self.buffer = self._prepare_buffer(phase, "failover/recovered", self.all_ranks)
        if self.is_replacement:
            print(f"rank {self.rank} pid={os.getpid()}: replacement joined", flush=True)
        else:
            self._check_stale_handle(*stale_round)

        self._run_round(self.buffer, self.all_ranks, phase)
        self.buffer.destroy()
        self._report_pass(phase, self.all_ranks)


def worker(rank, is_replacement, config):
    # Keep the process entry point at module scope for multiprocessing spawn.
    FailoverWorker(rank, is_replacement, config).run()


class FailoverLauncher:
    """Host the rendezvous store on node 0 and supervise local rank processes."""

    def __init__(self, config):
        import multiprocessing as mp

        self.config = config
        self.store = open_store(config, server=config.node_idx == 0)
        self.ctx = mp.get_context("spawn")
        first_rank_idx = config.node_idx * config.ranks_per_node
        self.local_ranks = range(first_rank_idx, first_rank_idx + config.ranks_per_node)
        self.workers = {}
        self.failure_round_keys = [
            f"{Round.FAILURE.label}/{rank}" for rank in config.survivors
        ]
        self.done_keys = [f"done/{node}" for node in range(config.nodes)]
        self.error_keys = [f"error/{node}" for node in range(config.nodes)]
        self.deadline = time.monotonic() + config.timeout_s

    def _start(self, rank, is_replacement=False):
        name = f"rank{rank}" + ("-replacement" if is_replacement else "")
        proc = self.ctx.Process(
            target=worker, args=(rank, is_replacement, self.config), name=name
        )
        proc.start()
        self.workers[rank] = (proc, is_replacement)

    def _reap_workers(self):
        for rank, (proc, is_replacement) in list(self.workers.items()):
            exitcode = proc.exitcode
            if exitcode is None:
                continue
            expects_crash = not is_replacement and rank in self.config.failed_ranks
            expected = CRASH_EXIT_CODE if expects_crash else 0
            if exitcode != expected:
                raise RuntimeError(
                    f"{proc.name} exited with {exitcode}, expected {expected}"
                )
            proc.join()
            del self.workers[rank]
            if expects_crash:
                self.store.set(f"exited/{rank}", "1")

    def _watch(self, complete, phase):
        """Wait for a checkpoint without resetting the overall timeout."""
        while True:
            for key in self.error_keys:
                if self.store.check([key]):
                    raise RuntimeError(self.store.get(key).decode())
            self._reap_workers()
            if complete():
                return
            if time.monotonic() >= self.deadline:
                raise TimeoutError(f"timed out waiting for {phase}")
            time.sleep(0.1)

    def _terminate_all(self):
        for proc, _ in self.workers.values():
            if proc.is_alive():
                proc.terminate()
        for proc, _ in self.workers.values():
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()

    def _report_buffer_setup_times(self):
        groups = (
            (Round.PRE_FAILURE, "make_buffer", range(self.config.world_size)),
            (Round.FAILURE, "make_buffer", self.config.survivors),
            (Round.AFTER_RECOVERY, "make_buffer", self.config.failed_ranks),
            (Round.AFTER_RECOVERY, "update_ep_member", self.config.survivors),
        )
        for phase, operation, members in groups:
            timings = self.store.multi_get(
                [f"buffer_setup_ns/{phase.label}/{rank}" for rank in members]
            )
            milliseconds = [int(timing) / 1_000_000 for timing in timings]
            per_rank = ", ".join(
                f"{rank}={duration:.3f}"
                for rank, duration in zip(members, milliseconds)
            )
            print(
                f"TIMING {phase.label} operation={operation} "
                f"max_ms={max(milliseconds):.3f} per_rank_ms=[{per_rank}]",
                flush=True,
            )

    def run(self):
        try:
            for rank in self.local_ranks:
                self._start(rank)
            self._watch(
                lambda: self.store.check(self.failure_round_keys),
                f"{Round.FAILURE.label} on all survivors",
            )

            for rank in self.local_ranks:
                if rank in self.config.failed_ranks:
                    self._start(rank, is_replacement=True)
            self._watch(
                lambda: not self.workers, "replacement and post-recovery checks"
            )
            self.store.set(self.done_keys[self.config.node_idx], "1")

            # Keep the TCPStore alive until every supervisor has finished.
            if self.config.node_idx == 0:
                self._watch(
                    lambda: self.store.check(self.done_keys), "all nodes to finish"
                )
                self._report_buffer_setup_times()
                print("All NCCL failover checks passed.", flush=True)
        except BaseException as error:
            try:
                self.store.set(
                    self.error_keys[self.config.node_idx],
                    f"node {self.config.node_idx}: {error}",
                )
            except Exception:
                pass
            raise
        finally:
            self._terminate_all()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--nodes", type=int, default=os.getenv("SLURM_NNODES", "2"))
    parser.add_argument("--ranks-per-node", type=int, default=8)
    parser.add_argument("--node-idx", type=int, default=os.getenv("SLURM_NODEID", "0"))
    parser.add_argument("--master-addr", default=os.getenv("MASTER_ADDR"))
    parser.add_argument("--port", type=int, default=29755)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument(
        "--failed-ranks",
        default=DEFAULT_FAILED_RANKS,
        help="comma-separated logical slots (default: %(default)s)",
    )
    config = parser.parse_args()

    # Sanity checks:
    if config.nodes != 2 or config.ranks_per_node != 8:
        parser.error("this test requires exactly 2 nodes and 8 ranks per node")
    if not 0 <= config.node_idx < config.nodes:
        parser.error("--node-idx must be 0 or 1")
    if not 1 <= config.port <= 65535 or config.timeout_s <= 0:
        parser.error("port must be in 1..65535 and timeout must be positive")
    config.world_size = config.nodes * config.ranks_per_node
    try:
        failed = tuple(sorted(int(rank) for rank in config.failed_ranks.split(",")))
    except ValueError:
        parser.error("--failed-ranks must be a comma-separated list of integers")
    full = set(range(config.world_size))
    failed_set = set(failed)
    if len(failed_set) != len(failed) or not failed_set <= full:
        parser.error("--failed-ranks must contain distinct ranks in 0..15")
    config.failed_ranks = failed
    config.survivors = tuple(sorted(full - failed_set))
    if any(
        sum(rank // config.ranks_per_node == node for rank in config.survivors)
        != SURVIVORS_PER_NODE
        for node in range(config.nodes)
    ):
        parser.error("--failed-ranks must leave 4 survivors on each node for Rail GIN")

    if not config.master_addr:
        if "SLURM_JOB_NODELIST" in os.environ:
            hosts = subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
                text=True,
            ).splitlines()
            if not hosts:
                parser.error("Slurm returned an empty host list")
            config.master_addr = hosts[0]
        else:
            parser.error("--master-addr is required for a two-node run outside Slurm")

    import torch

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < config.ranks_per_node
    ):
        parser.error(f"each node needs {config.ranks_per_node} visible CUDA GPUs")
    FailoverLauncher(config).run()


if __name__ == "__main__":
    main()
