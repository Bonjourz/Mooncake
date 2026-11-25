"""Four-GPU failover test for Mooncake EP elastic buffers.

Example: Ranks 0-3 start. Ranks 1 and 3 fail, and ranks 0 and 2 survive.

Scenario:
  a. Ranks 0-3 warm up dispatch and combine locally and connect to each other.
  b. Dispatch/combine works on ranks 0-3.
  c. Ranks 1 and 3 crash (os._exit, no cleanup).
  d. Ranks 0 and 2 run the next round still routing to ranks 1 and 3.
     GPU receive timeouts mask the failed ranks on each survivor, which
     detects the failures through query_mask_buffer().
  d2. Survivors serve with ranks 1 and 3 masked but still connected.
      The serving rounds must finish in less than timeout_ms total.
  e. Survivors disconnect ranks 1 and 3 and keep serving among themselves.
  f. New processes for ranks 1 and 3 warm up locally and reconnect to the
     survivors and to each other.
  g. Ranks 0-3 serve again.

The parent process hosts the TCPStore and restarts the failed ranks.
Survivors receive no crash notification from the parent. They identify the
failed ranks by reading query_mask_buffer() after the timeout round.

Use --failed-ranks to select which slots fail (default: 1,3). The other slots
survive by default (0,2 with the default failed ranks). Use --survivor-ranks
to select a subset of those slots. Rank r runs on CUDA device r.

Labels, by step. The lowest surviving rank prints "PASS <label>/<i>" for
rounds from _serve(), with i = 0 .. ROUNDS-1. The detect round prints
"PASS detect". The self-only warm-up does not print a PASS line.
  a. initial_connect  rendezvous after connect_ranks; no round.
  b. before_fail      ROUNDS rounds on ranks 0-3.
  d. detect           one round; ranks 0 and 2 still route to ranks 1 and 3.
  d2. after_detect    ROUNDS rounds on ranks 0 and 2; must finish in less
                       than timeout_ms in total.
  e. survivors_only   ROUNDS rounds on ranks 0 and 2 after disconnect.
  f. reconnect/start  rendezvous before connect_ranks; no round.
     rejoin           rendezvous after connect_ranks; no round.
  g. after_recover    ROUNDS rounds on ranks 0-3 with both replacements.
"""

import argparse
import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


CAPACITY = 4
EXPERTS_PER_RANK = 8
TOKENS = 16
HIDDEN = 4096
ROUNDS = 2
CRASH_EXIT_CODE = 17
STORE_TIMEOUT = timedelta(seconds=180)


def expected_mask(active):
    # Public convention: 1 means masked, 0 means active.
    return [0 if r in active else 1 for r in range(CAPACITY)]


class FailoverWorker:
    """One rank process: owns the elastic buffer and runs steps a-g."""

    def __init__(self, rank, is_replacement, port, timeout_ms, failed_ranks,
                 survivors):
        from mooncake.mooncake_ep_buffer import Buffer

        torch.cuda.set_device(rank)
        self.rank = rank
        self.is_replacement = is_replacement
        self.timeout_ms = timeout_ms
        self.failed_ranks = tuple(failed_ranks)
        self.survivors = tuple(survivors)
        self.all_ranks = tuple(sorted(self.survivors + self.failed_ranks))
        self.device = torch.device("cuda", rank)
        self.store = dist.TCPStore("127.0.0.1", port, None, False, STORE_TIMEOUT)
        self.tag = f"rank {rank}" + (" (replacement)" if is_replacement else "")

        self.buffer = Buffer.create_elastic(
            rank=rank,
            tcp_store_group=self.store,
            explicitly_destroy=True,
            timeout_ms=timeout_ms,
            metadata_timeout_ms=int(STORE_TIMEOUT.total_seconds() * 1000),
        )
        self.buffer.update_memory_buffers(
            num_ranks=CAPACITY,
            num_experts_per_rank=EXPERTS_PER_RANK,
            num_rdma_bytes=Buffer.get_low_latency_buffer_size_hint(
                TOKENS, HIDDEN, CAPACITY, CAPACITY * EXPERTS_PER_RANK
            ),
        )
        self.stats = torch.zeros(
            EXPERTS_PER_RANK, dtype=torch.int32, device=self.device
        )
        self.expected_stats = 0
        self.mask = torch.empty(CAPACITY, dtype=torch.int32, device=self.device)

    def _rendezvous(self, label, ranks):
        # Each label is used once, so an old key never satisfies a new wait.
        torch.cuda.synchronize()
        self.store.set(f"demo/{label}/{self.rank}", "ready")
        self.store.wait([f"demo/{label}/{r}" for r in ranks if r != self.rank],
                        STORE_TIMEOUT)

    def _read_mask(self):
        self.buffer.query_mask_buffer(self.mask)
        return self.mask.cpu().tolist()

    def _run_round(self, label, *, report=True):
        """One dispatch/combine round, using only what the mask says.

        routing: ranks unmasked before the round; token t goes to expert 0 of
                 routing[t % len(routing)].
        live:    ranks still unmasked after the round. A rank whose receive
                 timed out inside dispatch/combine is masked and drops out.
        """
        rank, tag, device, buffer = self.rank, self.tag, self.device, self.buffer
        before = self._read_mask()
        routing = [r for r, masked in enumerate(before) if not masked]

        x = (
            torch.arange(1, TOKENS + 1, device=device, dtype=torch.float32)
            + rank * 32
        ).to(torch.bfloat16).view(TOKENS, 1).expand(TOKENS, HIDDEN).contiguous()
        dest = [routing[t % len(routing)] for t in range(TOKENS)]
        topk = torch.tensor(
            [d * EXPERTS_PER_RANK for d in dest], dtype=torch.int64, device=device
        ).view(TOKENS, 1)
        weights = torch.ones((TOKENS, 1), dtype=torch.float32, device=device)

        recv_x, counts, handle, _, _ = buffer.dispatch(
            x, topk,
            num_max_dispatch_tokens_per_rank=TOKENS,
            num_experts=CAPACITY * EXPERTS_PER_RANK,
            cumulative_local_expert_recv_stats=self.stats,
            use_fp8=False,
        )
        # Identity expert.
        out, _, _ = buffer.combine(recv_x, topk, weights, handle, zero_copy=False)
        torch.cuda.synchronize()
        after = self._read_mask()
        if rank == 0 and before != after:
            print(f"{tag}: mask changed during serving, label={label}: before={before} "
                  f"after={after}", flush=True)

        live = [r for r, masked in enumerate(after) if not masked]
        assert set(live) <= set(routing), (label, tag, before, after)

        # Tokens sent to live ranks must round-trip exactly. Tokens sent to a
        # dead rank are lost for this round; the application would retry them.
        kept = torch.tensor([d in live for d in dest], device=device)
        torch.testing.assert_close(out[kept], x[kept], rtol=0, atol=0)

        # Every live sender uses the same routing, so expert 0 here receives
        # len(live) * (tokens routed to me) tokens; dead senders add zero.
        per_sender = sum(1 for d in dest if d == rank)
        expected_counts = [len(live) * per_sender] + [0] * (EXPERTS_PER_RANK - 1)
        assert counts.cpu().tolist() == expected_counts, (label, tag, counts)
        self.expected_stats += len(live) * per_sender
        assert self.stats.cpu().tolist() == (
            [self.expected_stats] + [0] * (EXPERTS_PER_RANK - 1)
        ), (label, tag, self.stats)

        if report and rank == min(self.survivors):
            lost = TOKENS - int(kept.sum())
            print(f"PASS {label} routing={routing} live={live} "
                  f"lost_tokens={lost}", flush=True)

    def _serve(self, phase):
        for i in range(ROUNDS):
            self._run_round(f"{phase}/{i}")

    def _get_and_verify_mask(self, label, active):
        # Test-only check of what Mooncake put in the mask; not used to serve.
        mask = self._read_mask()
        assert mask == expected_mask(active), (label, self.tag, mask)
        return mask

    def run(self):
        # First-use compilation can take longer than the GPU receive timeout.
        # Use the normal round shape so both kernels are ready before peers wait.
        self._get_and_verify_mask("warmup/start", (self.rank,))
        self._run_round("warmup", report=False)
        self._get_and_verify_mask("warmup/done", (self.rank,))

        if not self.is_replacement:
            # a. Initial bring-up of ranks 0-3.
            self.buffer.connect_ranks([r for r in self.all_ranks if r != self.rank])
            self._rendezvous("initial_connect", self.all_ranks)
            self.buffer.barrier()

            # b. Normal serving on ranks 0-3.
            self._serve("before_fail")
            self._get_and_verify_mask("before_fail", self.all_ranks)
            # Finish the baseline on every rank before any worker exits.
            self._rendezvous("before_fail/done", self.all_ranks)

            # c. The selected ranks crash without cleanup.
            if self.rank in self.failed_ranks:
                print(f"{self.tag}: crashing now", flush=True)
                os._exit(CRASH_EXIT_CODE)

            # d. Survivors still route to failed ranks until receive timeouts mask them.
            t0 = time.monotonic()
            self._run_round("detect")
            mask = self._read_mask()
            failed = [r for r, masked in enumerate(mask) if masked]
            print(f"{self.tag}: detected failed ranks {failed} after "
                  f"{time.monotonic() - t0:.1f}s, mask={mask}", flush=True)
            self._get_and_verify_mask("detect", self.survivors)

            # d2. Survivors serve with failed ranks masked but still connected.
            # Any round that waited for a timeout would take >= timeout_ms.
            t0 = time.monotonic()
            self._serve("after_detect")
            elapsed = time.monotonic() - t0
            self._get_and_verify_mask("after_detect", self.survivors)
            assert elapsed < self.timeout_ms / 1000, (self.tag, elapsed)
            print(f"{self.tag}: after_detect took {elapsed:.3f}s, "
                  f"no timeout", flush=True)

            # e. Survivors update their state and keep serving.
            self.buffer.disconnect_ranks(failed)
            self._serve("survivors_only")
            self._get_and_verify_mask("survivors_only", self.survivors)
            self.store.set(f"demo/survivors_only/done/{self.rank}", "done")

            # f. Reconnect every failed slot once the replacements are ready.
            self._rendezvous("reconnect/start", self.all_ranks)
            start_time = time.monotonic()
            self.buffer.connect_ranks(failed)
            end_time = time.monotonic()
        else:
            # f. Each replacement connects to the survivors and its fellow replacements.
            self._rendezvous("reconnect/start", self.all_ranks)
            start_time = time.monotonic()
            self.buffer.connect_ranks([r for r in self.all_ranks if r != self.rank])
            end_time = time.monotonic()
        print(f"{self.tag}: reconnect (connect_ranks) took "
              f"{(end_time - start_time) * 1000:.1f} ms", flush=True)
        self._rendezvous("rejoin", self.all_ranks)
        self.buffer.barrier()

        # g. Normal serving on ranks 0-3 again.
        self._serve("after_recover")
        self._get_and_verify_mask("after_recover", self.all_ranks)

        self._rendezvous("destroy/start", self.all_ranks)
        self.buffer.destroy()
        self._rendezvous("destroy/done", self.all_ranks)


def worker(rank, is_replacement, port, timeout_ms, failed_ranks, survivors):
    # Process entry point; must stay at module level so spawn can pickle it.
    FailoverWorker(rank, is_replacement, port, timeout_ms, failed_ranks,
                   survivors).run()


class FailoverLauncher:
    """Host the TCPStore, watch workers, and restart failed ranks."""

    def __init__(self, port, timeout_ms, failed_ranks, survivors):
        self.port = port
        self.timeout_ms = timeout_ms
        self.failed_ranks = tuple(failed_ranks)
        self.survivors = tuple(survivors)
        self.all_ranks = tuple(sorted(self.survivors + self.failed_ranks))
        # The store coordinates the workers and the parent process.
        self.store = dist.TCPStore("127.0.0.1", port, None, True, STORE_TIMEOUT,
                                   wait_for_workers=False)
        self.ctx = mp.get_context("spawn")
        self.procs = {}

    def _start(self, rank, is_replacement):
        proc = self.ctx.Process(target=worker,
                                args=(rank, is_replacement, self.port, self.timeout_ms,
                                      self.failed_ranks, self.survivors),
                                name=f"rank{rank}" + ("-replacement" if is_replacement else ""))
        proc.start()
        self.procs[f"rank{rank}"] = proc
        return proc

    def _watch(self, done, label, timeout_s=600, *, expected_exits=None):
        """Wait for done() and reject unexpected worker exit codes."""
        expected_exits = expected_exits or {}
        deadline = time.monotonic() + timeout_s
        while True:
            finished = done()
            for name, proc in self.procs.items():
                exitcode = proc.exitcode
                expected = expected_exits.get(name, 0)
                if exitcode is not None and exitcode != expected:
                    raise RuntimeError(f"{name} exited with {exitcode}, expected {expected} ({label})")
            if finished:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for {label}")
            time.sleep(0.2)

    def _survivors_done(self):
        return self.store.check([f"demo/survivors_only/done/{r}" for r in self.survivors])

    def _all_exited(self):
        return all(p.exitcode is not None for p in self.procs.values())

    def _terminate_all(self):
        for proc in self.procs.values():
            if proc.is_alive():
                proc.terminate()
        for proc in self.procs.values():
            proc.join(10)

    def run(self):
        try:
            for rank in self.all_ranks:
                self._start(rank, False)

            # Keep the old workers tracked until they all exit as intended.
            crashed = {f"rank{rank}": self.procs[f"rank{rank}"] for rank in self.failed_ranks}
            self._watch(
                lambda: all(proc.exitcode is not None for proc in crashed.values()),
                "planned crashes",
                expected_exits=dict.fromkeys(crashed, CRASH_EXIT_CODE),
            )
            for name in crashed:
                del self.procs[name]
            print(f"ranks {list(self.failed_ranks)} are gone", flush=True)

            self._watch(self._survivors_done, "survivor-only serving")
            for rank in self.failed_ranks:
                print(f"restarting rank {rank}", flush=True)
                self._start(rank, True)
            self._watch(self._all_exited, "final rounds")
        except BaseException as error:
            self._terminate_all()
            print(f"FAILED: {error}", file=sys.stderr, flush=True)
            sys.exit(1)
        print("All failover checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--port", type=int, default=29754)
    parser.add_argument("--timeout-ms", type=int, default=5000,
                        help="GPU receive timeout used to detect failed ranks")
    parser.add_argument("--failed-ranks",
                        help="comma-separated ranks to crash and replace (default: %(default)s)")
    parser.add_argument("--survivor-ranks",
                        help="comma-separated ranks that stay up (default: all other slots, normally 0,2)")
    args = parser.parse_args()

    def parse_ranks(value, option):
        try:
            ranks = tuple(sorted(int(rank) for rank in value.split(",")))
        except ValueError:
            parser.error(f"{option} must be a nonempty comma-separated list of integers.")
        if len(set(ranks)) != len(ranks):
            parser.error(f"{option} must contain distinct ranks.")
        if any(not 0 <= rank < CAPACITY for rank in ranks):
            parser.error(f"{option} ranks must be in 0..{CAPACITY - 1}.")
        return ranks

    failed_ranks = parse_ranks(args.failed_ranks, "--failed-ranks")
    survivors = (
        tuple(rank for rank in range(CAPACITY) if rank not in failed_ranks)
        if args.survivor_ranks is None
        else parse_ranks(args.survivor_ranks, "--survivor-ranks")
    )
    if not survivors or set(failed_ranks) & set(survivors):
        parser.error("At least one survivor is required, and survivor ranks must exclude failed ranks.")
    all_ranks = tuple(sorted(survivors + failed_ranks))

    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(all_ranks):
        parser.error(f"This test requires {max(all_ranks) + 1} visible CUDA GPUs.")
    for a in all_ranks:
        for b in all_ranks:
            if a != b and not torch.cuda.can_device_access_peer(a, b):
                parser.error("Select GPUs with mutual CUDA P2P access for the participating ranks.")

    FailoverLauncher(args.port, args.timeout_ms, failed_ranks, survivors).run()


if __name__ == "__main__":
    main()
