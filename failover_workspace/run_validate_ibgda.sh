#!/usr/bin/env bash
# Run on a compute node after building Mooncake EP.
set -euo pipefail

# By default, ranks 1 and 3 fail and ranks 0 and 2 survive.
FAILED_RANKS="${FAILED_RANKS-1,3}"
SURVIVOR_RANKS="${SURVIVOR_RANKS-0,2}"
rank_args=(--failed-ranks "$FAILED_RANKS" --survivor-ranks "$SURVIVOR_RANKS")

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mooncake_repo="${MOONCAKE_REPO:-/home/ubuntu/bojunz/Mooncake}"
build_dir="${BUILD_DIR:-$mooncake_repo/build}"
stage="${STAGE:-$build_dir/validate_elastic_stage}"
ep_dir="$build_dir/mooncake-ep/src"

mkdir -p "$stage"
cp -r "$mooncake_repo/python/mooncake" "$stage/"
cp "$ep_dir/"_ep.cpython-*.so "$ep_dir/libmooncake_ep_device.so" "$stage/mooncake/"

export PYTHONPATH="$stage${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/ubuntu/.local}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="/home/ubuntu/bojunz/nccl/build/lib:/home/ubuntu/bojunz/tools/mooncake-deps/lib:$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$script_dir"
exec python3 validate_ibgda_failover.py "${rank_args[@]}" "$@"
