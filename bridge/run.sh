#!/usr/bin/env bash
# 启动 hw26_sim_bridge（假定模拟器已在另一个终端跑起来）
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bin="$here/build/hw26_sim_bridge"
[ -x "$bin" ] || { echo "还没编译，先跑 ./build.sh"; exit 1; }
echo "提示：模拟器应在另一个终端运行"
echo "      cd bevy_robomaster_simulator && ./scripts/run_reborn.sh -- --mode full"
echo "      首次使用先跑一次 sudo python3 scripts/setup-virtual-gimbal.py"
echo
exec "$bin" --config "$here/config/bridge.yaml" "$@"
