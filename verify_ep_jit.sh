#!/bin/bash
#
# [Remove me] Verify the JIT dispatch path end to end (JIT roadmap step 3.1).
#
#   ./verify_ep_jit.sh              # clear cache, run, analyze
#   ./verify_ep_jit.sh --keep-cache # keep the JIT cache (exercise the warm path)
#   ./verify_ep_jit.sh --no-rerun   # skip the second (cache-hit) run
#   ./verify_ep_jit.sh --kill-stale # reap survivors of an interrupted run
#   ./verify_ep_jit.sh --pg-backend mooncake   # default is nccl; see below
#   ./verify_ep_jit.sh --timeout 300           # per-run wall clock, default 900s
#
# The process-group backend is NOT what is under test here -- it only has to
# get the ranks talking so dispatch runs. It defaults to nccl because the
# mooncake backend needs RC queue pairs, which EFA-class fabrics reject with
# "Failed to create QP: Operation not supported", and the link warmup then
# retries until the timeout without ever reaching dispatch.
#
# Dispatch to a compute node with ~/command.sh:
#   ../command.sh ./verify_ep_jit.sh
#
# What it checks, in order:
#   1. build artifacts exist          -> otherwise staging silently half-copies
#   2. JIT cache is cold              -> otherwise nvcc is never actually called
#   3. the EP correctness test passes -> dispatch produces right numbers
#   4. nvcc really compiled           -> compile_begin/succeeded in the log
#   5. both variants got generated    -> fp8 AND bf16, not just one
#   6. a second run hits the cache    -> warm path works, overhead drops
#
# Exit code is non-zero if any check fails. Every check prints PASS/FAIL, and
# the analysis always runs even when the test itself fails -- a failed test
# with a missing fp8 variant is a very different bug from a failed test with
# both variants present.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
STAGE="${STAGE:-$BUILD_DIR/jit_verify_stage}"
LOG_DIR="${LOG_DIR:-/tmp/ep_jit_verify}"
JIT_CACHE="${MOONCAKE_EP_JIT_CACHE_DIR:-/tmp/mooncake_ep/jit}"

DEPS_PREFIX="${DEPS_PREFIX:-/home/ubuntu/bojunz/tools/mooncake-deps}"
NCCL_LIB="${NCCL_LIB:-/home/ubuntu/bojunz/nccl-extensions/third_party/nccl/build/lib}"

TEST="${TEST:-$REPO_ROOT/python/tests/ep/test_mooncake_ep.py}"

KEEP_CACHE=0
RERUN=1
# The mooncake PG backend needs RC queue pairs, which EFA-class fabrics do not
# support; there the link warmup retries forever and dispatch is never reached.
# run_ep_benchmark.py defaults to nccl for the same reason.
PG_BACKEND="${PG_BACKEND:-nccl}"
TIMEOUT="${TIMEOUT:-900}"
KILL_STALE=0

while [ $# -gt 0 ]; do
    arg="$1"
    case "$arg" in
        --keep-cache) KEEP_CACHE=1 ;;
        --no-rerun)   RERUN=0 ;;
        --kill-stale) KILL_STALE=1 ;;
        --pg-backend) shift; PG_BACKEND="${1:?--pg-backend needs a value}" ;;
        --pg-backend=*) PG_BACKEND="${arg#*=}" ;;
        --timeout)    shift; TIMEOUT="${1:?--timeout needs a value}" ;;
        -h|--help)
            awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' \
                "${BASH_SOURCE[0]}"
            exit 0 ;;
        *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
    shift
done

# Exported early: the import self-check in step 2 needs them just as much as
# the test run in step 4 does.
export MOONCAKE_EP_JIT_LOG=1
export MOONCAKE_EP_JIT_CACHE_DIR="$JIT_CACHE"
export MOONCAKE_EP_TEST_PG_BACKEND="$PG_BACKEND"

# ep_test_utils.init_dist defaults the rendezvous to port 8361. Two runs back
# to back -- which is exactly what this script does -- can collide with the
# previous store still in TIME_WAIT. Ask the kernel for a free one each time.
PORT_FIXED=0
[ -n "${MASTER_PORT:-}" ] && PORT_FIXED=1
pick_port() {
    [ "$PORT_FIXED" -eq 1 ] && return
    MASTER_PORT="$(python3 -c 'import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()')"
    export MASTER_PORT
}
pick_port
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/ubuntu/.local}"
export LD_LIBRARY_PATH="$NCCL_LIB:$DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FAILURES=0
pass() { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run1.log"
RERUN_LOG="$LOG_DIR/run2.log"

# ---------------------------------------------------------------- preflight
step "1. Build artifacts"

# run_ep_bench.sh copies these five into the stage. A missing one makes cp
# fail mid-way and leaves an importable-but-broken package, which surfaces
# much later as a confusing ImportError.
ARTIFACTS=(
    "$BUILD_DIR"/mooncake-ep/src/_ep.cpython-*.so
    "$BUILD_DIR"/ep_pg_staging/pg_*.cpython-*.so
    "$BUILD_DIR"/ep_pg_staging/libmooncake_ep_device.so
    "$BUILD_DIR"/ep_pg_staging/libmooncake_pg_device.so
    "$BUILD_DIR"/mooncake-pg/src/libmooncake_pg.so
)
missing=0
for pattern in "${ARTIFACTS[@]}"; do
    # shellcheck disable=SC2086  # deliberate glob expansion
    if ! compgen -G "$pattern" >/dev/null; then
        fail "missing: $pattern"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo
    echo "Build first:  ./make_ep.sh -DUSE_NCCL_DEVICE=ON"
    exit 1
fi
pass "all five build artifacts present"

if [ ! -f "$TEST" ]; then
    fail "test not found: $TEST"
    exit 1
fi

# Survivors of an interrupted run keep the rendezvous port bound and keep
# writing into the stage dir, which shows up much later as EADDRINUSE and as
# "rm: cannot remove ...: Directory not empty".
# spawn workers run as "python3 -c ... spawn_main", so match both forms.
STALE="$(pgrep -f "[p]ython3 .*$(basename "$TEST")|[s]pawn_main" || true)"
if [ -n "$STALE" ]; then
    if [ "$KILL_STALE" -eq 1 ]; then
        warn "killing stale test processes: $(echo "$STALE" | tr '\n' ' ')"
        # shellcheck disable=SC2086  # want word splitting here
        kill -9 $STALE 2>/dev/null || true
        sleep 2
    else
        fail "stale test processes still running: $(echo "$STALE" | tr '\n' ' ')"
        echo "        They hold the rendezvous port and the stage dir."
        echo "        Rerun with --kill-stale, or:  kill -9 $(echo "$STALE" | tr '\n' ' ')"
        exit 1
    fi
fi
pass "no stale test processes"

# ------------------------------------------------------------------ staging
step "2. Assemble importable mooncake package"

rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$REPO_ROOT/python/mooncake" "$STAGE/" || exit 1
for pattern in "${ARTIFACTS[@]}"; do
    cp $pattern "$STAGE/mooncake/" || exit 1
done

# mooncake.pg is a thin shim that forwards to the torch-version-suffixed
# extension (mooncake.pg_2_x_y). It ships in the wheel tree, NOT in
# python/mooncake/, so copying python/mooncake alone leaves ep_test_utils.py
# failing at "import mooncake.pg". run_ep_bench.sh gets away without it
# because the benchmark only touches mooncake.ep.
cp "$REPO_ROOT/mooncake-wheel/mooncake/pg.py" "$STAGE/mooncake/" || exit 1

export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"
pass "staged to $STAGE"

# Fail here, with the offending module named, rather than 200 lines into a
# test traceback.
for mod in mooncake.ep mooncake.pg mooncake.mooncake_ep_buffer; do
    if err=$(python3 -c "import $mod" 2>&1); then
        pass "import $mod"
    else
        fail "import $mod"
        echo "$err" | tail -3 | sed 's/^/        /'
    fi
done
if [ "$FAILURES" -ne 0 ]; then
    echo
    echo "Staging is incomplete -- the test cannot run. Fix the above first."
    exit 1
fi

# -------------------------------------------------------------- cache state
step "3. JIT cache state"

if [ "$KEEP_CACHE" -eq 1 ]; then
    warn "--keep-cache: leaving $JIT_CACHE in place"
    warn "the compile-chain check below will be skipped (expect cache hits)"
else
    if [ -d "$JIT_CACHE" ]; then
        echo "  clearing $JIT_CACHE ($(find "$JIT_CACHE" -maxdepth 1 -mindepth 1 -type d | wc -l) variant dirs)"
    fi
    rm -rf "$JIT_CACHE"
    pass "cache cleared -- nvcc will be exercised for real"
fi

# ---------------------------------------------------------------- execution
step "4. Run the EP correctness test"

echo "  log: $RUN_LOG"
echo "  pg backend: $PG_BACKEND   timeout: ${TIMEOUT}s"
echo
# Run in its own process group so a stall or timeout can kill the spawn
# workers too, and watch the log for progress: a rank deadlocked in a
# collective never exits, so only the wall-clock timeout would catch it.
STALL="${STALL:-90}"
STALLED=0
setsid timeout -k 5 "$TIMEOUT" python3 "$TEST" >"$RUN_LOG" 2>&1 &
TEST_PID=$!
tail -n +1 -f "$RUN_LOG" --pid="$TEST_PID" &
TAIL_PID=$!
last=-1; idle=0
while kill -0 "$TEST_PID" 2>/dev/null; do
    sleep 5
    cur=$(stat -c %s "$RUN_LOG" 2>/dev/null || echo 0)
    if [ "$cur" -eq "$last" ]; then idle=$((idle + 5)); else idle=0; last=$cur; fi
    if [ "$idle" -ge "$STALL" ]; then
        STALLED=1
        kill -TERM -- "-$TEST_PID" 2>/dev/null; sleep 3
        kill -KILL -- "-$TEST_PID" 2>/dev/null
        break
    fi
done
wait "$TEST_PID" 2>/dev/null; TEST_RC=$?
wait "$TAIL_PID" 2>/dev/null
echo

if [ "$STALLED" -eq 1 ]; then
    fail "no log output for ${STALL}s -- a rank is deadlocked; killed -- see $RUN_LOG"
    warn "if the log stops after 'IBGDA unavailable', that is the connect() /"
    warn "update_ep_member branch mismatch (test mocks a rebuilt rank 1), not a JIT problem."
elif [ "$TEST_RC" -eq 0 ]; then
    pass "test exited 0"
elif [ "$TEST_RC" -eq 124 ]; then
    fail "test timed out after ${TIMEOUT}s -- see $RUN_LOG"
else
    fail "test exited $TEST_RC -- see $RUN_LOG"
fi

# A transport that never comes up looks like a dispatch failure in the summary
# below (no compile, no variants) but has nothing to do with the JIT path.
# Name it explicitly so the next run does not chase the wrong thing.
if grep -q "Failed to create QP" "$RUN_LOG"; then
    echo
    warn "the fabric rejected QP creation -- this is a transport problem,"
    warn "not a JIT one: dispatch was never reached."
    warn "check with:  ibv_devinfo | grep -iE 'hca_id|transport'"
    if [ "$PG_BACKEND" != "nccl" ]; then
        warn "retry with:  $0 --pg-backend nccl"
    fi
fi

# ----------------------------------------------------------------- analysis
step "5. Compile chain"

if [ "$KEEP_CACHE" -eq 1 ]; then
    warn "skipped (--keep-cache)"
elif grep -q "compile_succeeded" "$RUN_LOG"; then
    pass "nvcc ran and succeeded"
    grep -oE "nvcc_process_sec=[0-9.]+|jit_overhead_sec=[0-9.]+" "$RUN_LOG" \
        | sort -u | sed 's/^/        /'
else
    fail "no compile_succeeded in log -- nvcc was never invoked"
    echo "        (a stale cache elsewhere? check MOONCAKE_EP_JIT_CACHE_DIR)"
    grep -E "disk_cache=|compile_(begin|failed)|JIT " "$RUN_LOG" | head -20 \
        | sed 's/^/        /'
fi

if grep -q "compile_failed" "$RUN_LOG"; then
    fail "a compile_failed appears in the log"
    # nvcc's own stderr is kept next to the source it failed on.
    find "$JIT_CACHE" -name compile.log -newer "$STAGE" 2>/dev/null \
        | head -3 | while read -r f; do
            echo "        --- $f ---"
            sed 's/^/        /' "$f" | head -20
        done
fi

step "6. Generated variants"

mapfile -t SOURCES < <(find "$JIT_CACHE" -name kernel.cu 2>/dev/null | sort)
echo "  found ${#SOURCES[@]} generated source(s) under $JIT_CACHE"

if [ "${#SOURCES[@]}" -eq 0 ]; then
    fail "no kernel.cu generated at all"
else
    for f in "${SOURCES[@]}"; do
        printf '        %s\n' "$(grep -h "dispatch_kernel_impl<" "$f" | sed 's/^ *//')"
    done

    # The test drives both the fp8 and the bf16 path, so a correct run must
    # produce both template instantiations. Seeing only one means half the
    # test never reached dispatch.
    have_fp8=0; have_bf16=0
    grep -qh "dispatch_kernel_impl<true,"  "${SOURCES[@]}" && have_fp8=1
    grep -qh "dispatch_kernel_impl<false," "${SOURCES[@]}" && have_bf16=1

    [ "$have_bf16" -eq 1 ] && pass "bf16 variant generated" \
                           || fail "no bf16 variant (<false, ...>)"
    [ "$have_fp8" -eq 1 ]  && pass "fp8 variant generated" \
                           || fail "no fp8 variant (<true, ...>) -- fp8 path not exercised"

    # Guard the generator's output shape. These are the three things that can
    # only blow up at runtime: wrong entry name, dropped extern "C", wrong
    # launch bounds arithmetic.
    for f in "${SOURCES[@]}"; do
        grep -q 'extern "C" __global__' "$f" || fail "$f: missing extern \"C\" __global__"
        grep -q "mooncake_ep_jit_dispatch_kernel" "$f" || fail "$f: wrong entry name"
        grep -q "EP_LAUNCH_BOUNDS(8 \* 4 \* 32, 1)" "$f" || fail "$f: unexpected launch bounds"
    done
    pass "entry name / extern \"C\" / launch bounds shape OK"
fi

step "7. Assertions and errors in the log"

if grep -qiE "Traceback|AssertionError|CUDA error|EPException" "$RUN_LOG"; then
    fail "error signatures found:"
    grep -inE "Traceback|AssertionError|CUDA error|EPException" "$RUN_LOG" \
        | head -10 | sed 's/^/        /'
else
    pass "no traceback / assertion / CUDA error in log"
fi

# ---------------------------------------------------------------- warm path
if [ "$RERUN" -eq 1 ] && [ "$TEST_RC" -eq 0 ]; then
    step "8. Second run (warm cache)"
    echo "  log: $RERUN_LOG"
    pick_port
    timeout "$TIMEOUT" python3 "$TEST" >"$RERUN_LOG" 2>&1
    RERUN_RC=$?

    if [ "$RERUN_RC" -ne 0 ]; then
        fail "second run exited $RERUN_RC -- see $RERUN_LOG"
    elif grep -q "compile_begin" "$RERUN_LOG"; then
        fail "recompiled on the warm run -- disk cache is not being reused"
    else
        pass "no recompile: disk cache reused"
        echo "        cold: $(grep -oE 'jit_overhead_sec=[0-9.]+' "$RUN_LOG"   | sort -u | tr '\n' ' ')"
        echo "        warm: $(grep -oE 'jit_overhead_sec=[0-9.]+' "$RERUN_LOG" | sort -u | tr '\n' ' ')"
    fi
elif [ "$RERUN" -eq 1 ]; then
    step "8. Second run (warm cache)"
    warn "skipped: first run failed"
fi

# ------------------------------------------------------------------ verdict
step "Verdict"
if [ "$FAILURES" -eq 0 ]; then
    printf '  \033[32mall checks passed\033[0m -- JIT roadmap step 3.1 verified\n'
    echo "  logs: $RUN_LOG"
    exit 0
fi
printf '  \033[31m%d check(s) failed\033[0m\n' "$FAILURES"
echo "  logs: $RUN_LOG"
echo "  generated sources: $JIT_CACHE/*/kernel.cu"
exit 1
