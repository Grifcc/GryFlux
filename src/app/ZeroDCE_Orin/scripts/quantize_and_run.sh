#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${APP_ROOT}/../../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
APP_BIN="${BUILD_DIR}/src/app/ZeroDCE_Orin/zero_dce_app"

ONNX_PATH="${ONNX_PATH:-/workspace/zjx/model/ZeroDCE_static640.onnx}"
DATASET_DIR="${DATASET_DIR:-/workspace/zjx/data/DICM}"
ENGINE_PATH="${ENGINE_PATH:-${APP_ROOT}/3rdparty/models/ZeroDCE_static640_int8.engine}"
CACHE_PATH="${CACHE_PATH:-${APP_ROOT}/3rdparty/models/ZeroDCE_static640_int8.cache}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/zjx/output/zero_dce_int8}"
INPUT_SHAPE="${INPUT_SHAPE:-}"
MAX_IMAGES="${MAX_IMAGES:-128}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
FORCE_REBUILD_ENGINE="${FORCE_REBUILD_ENGINE:-0}"

mkdir -p "$(dirname "${ENGINE_PATH}")" "${OUTPUT_DIR}"

if [[ "${FORCE_REBUILD_ENGINE}" == "1" || ! -f "${ENGINE_PATH}" || ! -f "${CACHE_PATH}" ]]; then
  echo "[1/3] Building ZeroDCE INT8 engine"
  CMD=(
    python3 "${SCRIPT_DIR}/build_int8_engine.py"
    --onnx "${ONNX_PATH}"
    --dataset "${DATASET_DIR}"
    --engine "${ENGINE_PATH}"
    --cache "${CACHE_PATH}"
    --max-images "${MAX_IMAGES}"
    --workspace-mib "${WORKSPACE_MIB}"
  )
  if [[ -n "${INPUT_SHAPE}" ]]; then
    CMD+=(--input-shape "${INPUT_SHAPE}")
  fi
  "${CMD[@]}"
else
  echo "[1/3] Reusing existing ZeroDCE INT8 engine"
  echo "  Engine : ${ENGINE_PATH}"
  echo "  Cache  : ${CACHE_PATH}"
fi

echo "[2/3] Building zero_dce_app target"
cmake -S "${REPO_ROOT}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target zero_dce_orin -j"$(nproc)"

echo "[3/3] Running zero_dce_app"
echo "cd ${BUILD_DIR}"
echo "${APP_BIN} ${ENGINE_PATH} ${DATASET_DIR} ${OUTPUT_DIR}"
"${APP_BIN}" "${ENGINE_PATH}" "${DATASET_DIR}" "${OUTPUT_DIR}"

echo "Done."
echo "  Engine : ${ENGINE_PATH}"
echo "  Cache  : ${CACHE_PATH}"
echo "  Output : ${OUTPUT_DIR}"
