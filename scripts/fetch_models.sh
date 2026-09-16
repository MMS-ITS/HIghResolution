#!/usr/bin/env bash
# Download the Real-ESRGAN ONNX models into ./models/.
# The weights are ~69MB total and are deliberately not committed to git.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

fetch() {
  local name="$1" url="$2"
  if [[ -s "models/$name" ]]; then
    echo "==> models/$name already present, skipping"
    return
  fi
  echo "==> downloading $name"
  curl -fL --retry 3 --progress-bar -o "models/$name.part" "$url"
  mv "models/$name.part" "models/$name"
}

# Real-ESRGAN x4plus (RRDBNet-23) -- highest quality, ~64MB
fetch realesrgan_x4plus.onnx \
  "https://huggingface.co/anakhiu/realesrgan-onnx/resolve/main/realesrgan_x4plus.onnx"

# Real-ESRGAN general x4v3 (SRVGGNetCompact) -- ~10x faster, ~5MB
fetch realesr-general-x4v3.onnx \
  "https://huggingface.co/Heliosoph/realesrgan-onnx/resolve/main/realesr-general-x4v3.onnx"

echo
ls -lh models/
echo "Models ready."
