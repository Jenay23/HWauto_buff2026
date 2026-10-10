#!/usr/bin/env bash
# 把 HWauto_buff2026 的 buff.onnx 转成 OpenVINO IR（本桥用）
# 需要 OpenVINO 2024.x；不需要 TensorRT / CUDA / GPU
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ov="${OPENVINO_ROOT:-/opt/intel/openvino_2024.6.0}"
onnx="${1:-$here/../buff.onnx}"
out="${2:-$here/model/buff_ir.xml}"

[ -f "$onnx" ] || { echo "找不到 $onnx"; exit 1; }
mkdir -p "$(dirname "$out")"
source "$ov/setupvars.sh" >/dev/null 2>&1
export PYTHONPATH="$ov/python:${PYTHONPATH:-}"
python3 -m openvino.tools.ovc "$onnx" --output_model "$out" --compress_to_fp16
echo "OK -> $out (+ .bin)"
