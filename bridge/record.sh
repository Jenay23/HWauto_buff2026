#!/usr/bin/env bash
# 跑一次桥并把完整输出录到 logs/<时间戳>_<标签>.log
#
# 用法:
#   ./record.sh                    # 默认 --no-fire
#   ./record.sh --fire             # 带开火
#   ./record.sh --fire big1        # 第 2 个参数是标签，写进文件名
#
# 附带的 [dual] / [det] / [bridge] 行都在日志里，便于事后用 grep 分析：
#   grep '\[dual\]' logs/*.log
#   grep 'st=1'     logs/*.log
set -u
cd "$(dirname "$0")"
tag="${2:-run}"
stamp="$(date +%Y%m%d_%H%M%S)"
out="logs/${stamp}_${tag}.log"
mkdir -p logs
kpts="${out%.log}.kpts.jsonl"
echo "[record] -> $out"
echo "[record] -> $kpts   (真值8点 vs 模型18点，每帧一行)"
# 桥通过环境变量决定真值日志路径 -> 和本次 log 同名同标签，自动保存、不覆盖历史
HW26_KPTS_LOG="$kpts" ./build/hw26_sim_bridge --config config/bridge.yaml "${1:---no-fire}" 2>&1 | tee "$out"
echo "[record] saved: $out ($(wc -l < "$out") lines)"
echo "[record] 真值日志: $kpts ($(wc -l < "$kpts" 2>/dev/null || echo 0) lines)"
echo "[record] 对比: python3 tools/kpts_compare.py $kpts"
