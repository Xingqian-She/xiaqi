#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source_dir="$PWD/third_party/minizero"
output_file="$PWD/build/minizero/go/minizero_go"
if [[ ! -f "$source_dir/CMakeLists.txt" ]]; then
    echo 'Run git submodule update --init -- third_party/minizero in the host checkout.' >&2
    exit 1
fi

# The upstream build uses a generated header to report the source revision.
revision=$(git -c safe.directory="$source_dir" -C "$source_dir" rev-parse HEAD)

# Compile on the container filesystem to avoid slow I/O on Windows bind mounts.
scratch_dir=$(mktemp -d /tmp/xiaqi-minizero.XXXXXX)
trap 'rm -rf -- "$scratch_dir"' EXIT
build_dir="$scratch_dir/build"
mkdir -p "$scratch_dir/source" "$build_dir/git_info"
cp -R -- "$source_dir/CMakeLists.txt" "$source_dir/minizero" "$scratch_dir/source/"
git -C "$scratch_dir/source" apply --ignore-space-change "$PWD/third_party/minizero-sp-commands.patch"
printf '#pragma once\n#define GIT_HASH "%s"\n#define GIT_SHORT_HASH "%s"\n' \
    "$revision" "${revision:0:7}" > "$build_dir/git_info/git_info.h"

torch_prefix=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')
cmake -S "$scratch_dir/source" -B "$build_dir" -G Ninja \
    -DGAME_TYPE=GO -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS_RELEASE='-O3 -DNDEBUG -g0' \
    -DCMAKE_PREFIX_PATH="$torch_prefix" -DPython_EXECUTABLE="$(command -v python)"
cmake --build "$build_dir" --target minizero --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
install -D -m 755 "$build_dir/minizero_go" "$output_file"
