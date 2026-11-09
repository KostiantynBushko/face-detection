#!/usr/bin/env bash
#
# fetch.sh -- download the ONNX model weights used by face-detection.
#
# The weights come from the OpenCV model zoo:
#   https://github.com/opencv/opencv_zoo
#
# They are NOT committed to this repository (see .gitignore): they are large
# binaries and are better treated as build artefacts.
#
# Usage:
#   ./models/fetch.sh            # fetch the default (FP32) models
#   ./models/fetch.sh --int8     # fetch the quantised models (~10 MB instead
#                                # of ~39 MB for the recogniser)
#
# Environment:
#   FACE_DETECTION_MODELS   Directory to install into (default: this directory)

set -euo pipefail

MODELS_DIR="${FACE_DETECTION_MODELS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

ZOO_BASE="https://github.com/opencv/opencv_zoo/raw/main/models"

# name|path|sha256
MODELS=(
  "face_detection_yunet_2023mar.onnx|face_detection_yunet/face_detection_yunet_2023mar.onnx|8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
)

if [[ "${1:-}" == "--int8" ]]; then
  MODELS+=(
    "face_detection_yunet_2023mar_int8.onnx|face_detection_yunet/face_detection_yunet_2023mar_int8.onnx|"
    "face_recognition_sface_2021dec_int8.onnx|face_recognition_sface/face_recognition_sface_2021dec_int8.onnx|2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a"
  )
else
  MODELS+=(
    "face_recognition_sface_2021dec.onnx|face_recognition_sface/face_recognition_sface_2021dec.onnx|0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
  )
fi

mkdir -p "$MODELS_DIR"

have_sha256sum=false
if command -v sha256sum >/dev/null 2>&1; then
  have_sha256sum=true
elif command -v shasum >/dev/null 2>&1; then
  have_sha256sum=true
fi

verify() {
  local file="$1" expected="$2"
  [[ -z "$expected" ]] && return 0
  if ! $have_sha256sum; then
    echo "  ! no sha256sum/shasum available, skipping verification" >&2
    return 0
  fi
  local actual
  actual="$(sha256sum "$file" 2>/dev/null | cut -d' ' -f1)"
  if [[ -z "$actual" ]]; then
    actual="$(shasum -a 256 "$file" 2>/dev/null | cut -d' ' -f1)"
  fi
  if [[ "$actual" != "$expected" ]]; then
    echo "  ! checksum mismatch for $(basename "$file")" >&2
    echo "    expected $expected" >&2
    echo "    actual   $actual" >&2
    return 1
  fi
}

echo "Installing models into: $MODELS_DIR"

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name path expected <<< "$entry"
  target="$MODELS_DIR/$name"

  if [[ -f "$target" ]]; then
    echo "  = $name (already present)"
    continue
  fi

  echo "  + $name"
  url="$ZOO_BASE/$path"

  if command -v curl >/dev/null 2>&1; then
    curl -fsSL --retry 3 -o "$target.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$target.part" "$url"
  else
    echo "  ! neither curl nor wget is available" >&2
    exit 1
  fi

  if ! verify "$target.part" "$expected"; then
    rm -f "$target.part"
    exit 1
  fi

  mv "$target.part" "$target"
done

echo "Done. Models are in: $MODELS_DIR"
