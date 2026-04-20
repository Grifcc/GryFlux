#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRDPARTY_ROOT="${APP_ROOT}/3rdparty"

OPENCV_INCLUDE_SRC="${OPENCV_INCLUDE_SRC:-/usr/include/opencv4}"
OPENCV_LIB_SRC="${OPENCV_LIB_SRC:-/usr/lib/aarch64-linux-gnu}"
CUDA_INCLUDE_SRC="${CUDA_INCLUDE_SRC:-/usr/local/cuda/include}"
CUDA_LIB_SRC="${CUDA_LIB_SRC:-/usr/local/cuda/lib64}"
TENSORRT_INCLUDE_SRC="${TENSORRT_INCLUDE_SRC:-/usr/include/aarch64-linux-gnu}"
TENSORRT_LIB_SRC="${TENSORRT_LIB_SRC:-/usr/lib/aarch64-linux-gnu}"

link_latest_so() {
  local lib_root="$1"
  local stem="$2"
  local link_path="$3"
  local candidates=("${lib_root}/lib${stem}.so" "${lib_root}/lib${stem}.so".*)

  if ((${#candidates[@]} == 0)); then
    echo "Missing OpenCV library for ${stem} under ${lib_root}" >&2
    exit 1
  fi

  local target
  target="$(printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1)"
  ln -sfn "${target}" "${link_path}"
}

mkdir -p \
  "${THIRDPARTY_ROOT}/opencv/lib" \
  "${THIRDPARTY_ROOT}/cuda" \
  "${THIRDPARTY_ROOT}/tensorrt" \
  "${THIRDPARTY_ROOT}/models"

ln -sfn "${OPENCV_INCLUDE_SRC}" "${THIRDPARTY_ROOT}/opencv/include"
ln -sfn "${CUDA_INCLUDE_SRC}" "${THIRDPARTY_ROOT}/cuda/include"
ln -sfn "${CUDA_LIB_SRC}" "${THIRDPARTY_ROOT}/cuda/lib64"
ln -sfn "${TENSORRT_INCLUDE_SRC}" "${THIRDPARTY_ROOT}/tensorrt/include"
ln -sfn "${TENSORRT_LIB_SRC}" "${THIRDPARTY_ROOT}/tensorrt/lib"

link_latest_so "${OPENCV_LIB_SRC}" "opencv_core" \
  "${THIRDPARTY_ROOT}/opencv/lib/libopencv_core.so"
link_latest_so "${OPENCV_LIB_SRC}" "opencv_imgproc" \
  "${THIRDPARTY_ROOT}/opencv/lib/libopencv_imgproc.so"
link_latest_so "${OPENCV_LIB_SRC}" "opencv_imgcodecs" \
  "${THIRDPARTY_ROOT}/opencv/lib/libopencv_imgcodecs.so"

echo "resnet_Orin 3rdparty links are ready under: ${THIRDPARTY_ROOT}"
