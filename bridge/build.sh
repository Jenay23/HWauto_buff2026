#!/usr/bin/env bash
# 一键编译 hw26_sim_bridge（无需 TensorRT / CUDA / GPU）
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmake -S "$root" -B "$root/build" -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$root/build" --parallel "${BUILD_JOBS:-4}"
echo
echo "产物: $root/build/hw26_sim_bridge"
