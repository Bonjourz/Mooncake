#!/bin/bash
#
# Run the Mooncake EP dispatch/combine benchmark against the local build.
#
#   ./run_ep_bench.sh                          # k_hot, 8 ranks
#   ./run_ep_bench.sh --routing-mode uniform   # flags go to run_ep_benchmark.py
#   ./run_ep_bench.sh --config mooncake-ep/benchmarks/ep_benchmark/configs/cuda_zipfian.json
#
# Dispatch to a compute node with ~/command.sh:
#   ../command.sh ./run_ep_bench.sh
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"
DEPS_PREFIX="${DEPS_PREFIX:-/home/ubuntu/bojunz/tools/mooncake-deps}"
NCCL_LIB="${NCCL_LIB:-/home/ubuntu/bojunz/nccl-extensions/third_party/nccl/build/lib}"

# torch lives in /home/ubuntu/.local, but bashrc overrides HOME, which is what
# Python derives its user-site from.
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/ubuntu/.local}"

# torch needs libnccl.so.2 (absent from the compute-node image); the built
# extensions need the C++ deps.
export LD_LIBRARY_PATH="$NCCL_LIB:$DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Assemble an importable `mooncake` package: the sources plus the artifacts the
# build leaves in separate trees. Rebuilt every run so it never goes stale.
STAGE="${STAGE:-$BUILD_DIR/bench_stage}"
rm -rf "$STAGE"
mkdir -p "$STAGE"
cp -r "$REPO_ROOT/python/mooncake" "$STAGE/"
cp "$BUILD_DIR"/mooncake-ep/src/_ep.cpython-*.so \
   "$BUILD_DIR"/ep_pg_staging/pg_*.cpython-*.so \
   "$BUILD_DIR"/ep_pg_staging/libmooncake_ep_device.so \
   "$BUILD_DIR"/ep_pg_staging/libmooncake_pg_device.so \
   "$BUILD_DIR"/mooncake-pg/src/libmooncake_pg.so \
   "$STAGE/mooncake/"
export PYTHONPATH="$STAGE${PYTHONPATH:+:$PYTHONPATH}"

BENCH="$REPO_ROOT/mooncake-ep/benchmarks/ep_benchmark/run_ep_benchmark.py"
# --json-output is opened relative to CWD, so keep it absolute.
RESULTS="$(dirname "$BENCH")/results"
mkdir -p "$RESULTS"

if [ $# -gt 0 ]; then
    exec python3 "$BENCH" "$@"
fi

exec python3 "$BENCH" \
    --num-ranks 8 --num-experts 256 --hidden-size 7168 \
    --top-k 8 --num-tokens 1024 --dtype bf16 \
    --routing-mode k_hot --hot-experts 32 --hot-fraction 0.9 \
    --async-finish \
    --warmup-iters 20 --iters 100 \
    --pg-backend nccl \
    --json-output "$RESULTS/khot_8rank.json"
