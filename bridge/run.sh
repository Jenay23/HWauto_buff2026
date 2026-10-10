#!/usr/bin/env bash
# 启动 hw26_sim_bridge（假定模拟器已在另一个终端跑起来）
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bin="$here/build/hw26_sim_bridge"
[ -x "$bin" ] || { echo "还没编译，先跑 ./build.sh"; exit 1; }
echo "提示：模拟器应在另一个终端运行 (cd at_vision_simulator && cargo run --release)"
echo "      并在模拟器里按 F3 切视角、方向键对准符、Space 打一枪激活"
echo
exec "$bin" --config "$here/config/bridge.yaml" "$@"
