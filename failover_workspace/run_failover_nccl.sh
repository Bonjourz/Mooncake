#!/usr/bin/env bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=32
set -euo pipefail

if [[ ${SLURM_NNODES:-0} != 2 ]]; then
    echo "Run this from its checkout with ~/command.sh run_failover_nccl.sh" >&2
    exit 2
fi
RANKS_PER_NODE="${RANKS_PER_NODE-8}"
FAILED_RANKS="${FAILED_RANKS-4,5,6,7,12,13,14,15}"
TIMEOUT_S="${TIMEOUT_S-900}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
base=/home/ubuntu/bojunz
repo="${MOONCAKE_REPO:-$base/Mooncake}"
build_dir="${BUILD_DIR:-$repo/build}"
plugin_dir="${GIN_PLUGIN_DIR:-/shared/toolkits/aws-ofi-nccl-master/lib}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"

if [[ ! -r "$plugin_dir/libnccl-net.so" ]]; then
    echo "NCCL GIN plugin not found in $plugin_dir" >&2
    exit 1
fi

stage="$(mktemp -d)"
trap 'rm -rf -- "$stage"' EXIT
cp -r "$repo/python/mooncake" "$stage/"
cp "$build_dir/mooncake-ep/src/"_ep.cpython-*.so \
   "$build_dir/mooncake-ep/src/libmooncake_ep_device.so" "$stage/mooncake/"

export GLOG_minloglevel=1  # Mooncake C++ logs
export NCCL_DEBUG=WARN
export PYTHONPATH="$stage${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/ubuntu/.local}"
export PATH="/opt/slurm/bin:$cuda_home/bin:$PATH"
export LD_LIBRARY_PATH="$plugin_dir:/opt/amazon/efa/lib:$base/nccl/build/lib:$base/tools/mooncake-deps/lib:$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_GIN_TYPE="${NCCL_GIN_TYPE:-2}"
# This test briefly holds the old and replacement GIN contexts at once.
export MOONCAKE_EP_NCCL_GIN_CONTEXTS="${MOONCAKE_EP_NCCL_GIN_CONTEXTS:-17}"

python3 -u "$script_dir/validate_nccl_failover.py" \
    --ranks-per-node "$RANKS_PER_NODE" --failed-ranks "$FAILED_RANKS" \
    --timeout-s "$TIMEOUT_S" "$@"
