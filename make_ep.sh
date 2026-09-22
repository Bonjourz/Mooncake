#!/bin/bash
#
# Build Mooncake EP (and the PG extension it ships with).
#
#   ./make_ep.sh          # configure + build into ./build
#   ./make_ep.sh clean    # drop the build tree
#
# Extra args go to cmake, e.g. ./make_ep.sh -DUSE_NCCL_DEVICE=ON
# Dispatch to a compute node with ~/command.sh:
#   ../command.sh ./make_ep.sh
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build}"

# ~/.local/bin/cmake is a broken pip shim; prefer the system one.
CMAKE="${CMAKE:-/usr/bin/cmake}"
JOBS="${JOBS:-$(nproc)}"

# The C++ dependencies (yaml-cpp, gflags, glog, boost, grpc, ...) live in a
# micromamba prefix under /home/ubuntu/bojunz because /usr is node-local while
# /home is shared. Kept off PATH so the system python3 (the one with torch)
# still wins.
DEPS_PREFIX="${DEPS_PREFIX:-/home/ubuntu/bojunz/tools/mooncake-deps}"

# Some libs are linked by bare name (-lyaml-cpp), which CMAKE_PREFIX_PATH does
# not cover, so hand the prefix to the linker and the runtime loader directly.
export LIBRARY_PATH="$DEPS_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$DEPS_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# The EP/PG torch extensions are built by setup.py, which assembles its own -I
# list and ignores CMAKE_PREFIX_PATH.
export CPATH="$DEPS_PREFIX/include${CPATH:+:$CPATH}"

# nvcc is installed but not on PATH on the compute nodes.
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

if [ "$1" = "clean" ]; then
    rm -rf "$BUILD_DIR"
    echo "removed $BUILD_DIR"
    exit 0
fi

# WITH_EP compiles the extensions against the installed torch. Report the real
# import error: a present-but-unloadable torch looks the same as a missing one.
if ! torch_err=$(python3 -c "import torch" 2>&1); then
    echo "error: cannot import torch with $(command -v python3) ($(python3 -V 2>&1))." >&2
    echo "       WITH_EP builds the EP/PG extensions against the active torch." >&2
    echo "$torch_err" | tail -1 >&2
    exit 1
fi

"$CMAKE" -S "$REPO_ROOT" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_PREFIX_PATH="$DEPS_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWITH_EP=ON \
    -DUSE_CUDA=ON \
    -DWITH_STORE=OFF \
    -DWITH_STORE_RUST=OFF \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    "$@"

"$CMAKE" --build "$BUILD_DIR" -j "$JOBS"
