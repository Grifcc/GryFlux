#!/usr/bin/env bash
#
# GryFlux - AArch64 交叉编译脚本
#
# 依赖（宿主机）：
#   - cmake >= 3.16
#   - aarch64-linux-gnu 工具链在 PATH 中，或用 --toolchain-bin 指定 bin 目录
#
# 快速开始：
#   ./scripts/build-aarch64.sh --build-type Release
#
# 常用：
#   ./scripts/build-aarch64.sh --clean
#   ./scripts/build-aarch64.sh --install --prefix ./install/aarch64
#   ./scripts/build-aarch64.sh --sysroot /path/to/aarch64-sysroot
#   ./scripts/build-aarch64.sh --toolchain-bin /opt/gcc-aarch64/bin
#
# 兼容旧参数：
#   ./scripts/build-aarch64.sh clean
#   ./scripts/build-aarch64.sh release
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: ./scripts/build-aarch64.sh [options] [-- <extra cmake configure args...>]

Options:
  --build-type <Debug|Release|RelWithDebInfo|MinSizeRel>  Default: Debug
  --build-dir <path>                                     Default: build/aarch64
  --jobs <N>                                              Default: nproc
  --profile                                               Build with graph profiling compiled in (-DGRYFLUX_BUILD_PROFILING=1)
  --clean                                                 Remove build dir before configure
  --install                                               Run "cmake --install"
  --prefix <path>                                         Install prefix (needs --install)
  --sysroot <path>                                        Sets -DCMAKE_SYSROOT
  --toolchain-file <path>                                 Default: cmake/aarch64-toolchain.cmake
  --toolchain-prefix <prefix>                             Default: aarch64-linux-gnu
  --toolchain-bin <path>                                  Sets -DTOOLCHAIN_BIN_DIR for toolchain file
  -h, --help                                              Show this help

Shorthands (legacy):
  clean   == --clean
  release == --build-type Release

Examples:
  ./scripts/build-aarch64.sh --build-type Release
  ./scripts/build-aarch64.sh --clean --install --prefix ./install/aarch64
  ./scripts/build-aarch64.sh --sysroot /opt/aarch64-sysroot --build-type Release
  ./scripts/build-aarch64.sh -- --warn-uninitialized
EOF
}

BUILD_TYPE="Debug"
BUILD_DIR="${PROJECT_ROOT}/build/aarch64"
JOBS=""
DO_PROFILE=false
DO_CLEAN=false
DO_INSTALL=false
PREFIX=""
SYSROOT=""
TOOLCHAIN_FILE="${PROJECT_ROOT}/cmake/aarch64-toolchain.cmake"
TOOLCHAIN_PREFIX="aarch64-linux-gnu"
TOOLCHAIN_BIN_DIR=""
EXTRA_CMAKE_ARGS=()

while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    clean) DO_CLEAN=true; shift ;;
    release) BUILD_TYPE="Release"; shift ;;
    --build-type) BUILD_TYPE="${2:-}"; shift 2 ;;
    --build-dir) BUILD_DIR="${2:-}"; shift 2 ;;
    --jobs) JOBS="${2:-}"; shift 2 ;;
    --profile) DO_PROFILE=true; shift ;;
    --clean) DO_CLEAN=true; shift ;;
    --install) DO_INSTALL=true; shift ;;
    --prefix) PREFIX="${2:-}"; shift 2 ;;
    --sysroot) SYSROOT="${2:-}"; shift 2 ;;
    --toolchain-file) TOOLCHAIN_FILE="${2:-}"; shift 2 ;;
    --toolchain-prefix) TOOLCHAIN_PREFIX="${2:-}"; shift 2 ;;
    --toolchain-bin) TOOLCHAIN_BIN_DIR="${2:-}"; shift 2 ;;
    --) shift; EXTRA_CMAKE_ARGS+=("$@"); break ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS="8"
  fi
fi

if [[ ! -f "${TOOLCHAIN_FILE}" ]]; then
  echo "Toolchain file not found: ${TOOLCHAIN_FILE}" >&2
  exit 1
fi

if [[ -n "${SYSROOT}" && ! -d "${SYSROOT}" ]]; then
  echo "Sysroot does not exist: ${SYSROOT}" >&2
  exit 1
fi

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake not found in PATH" >&2
  exit 1
fi

toolchain_gcc="${TOOLCHAIN_PREFIX}-gcc"
toolchain_gxx="${TOOLCHAIN_PREFIX}-g++"
if [[ -n "${TOOLCHAIN_BIN_DIR}" ]]; then
  if [[ ! -x "${TOOLCHAIN_BIN_DIR}/${toolchain_gcc}" || ! -x "${TOOLCHAIN_BIN_DIR}/${toolchain_gxx}" ]]; then
    echo "Toolchain not found under --toolchain-bin: ${TOOLCHAIN_BIN_DIR}" >&2
    echo "Expected: ${TOOLCHAIN_BIN_DIR}/${toolchain_gcc} and ${TOOLCHAIN_BIN_DIR}/${toolchain_gxx}" >&2
    exit 1
  fi
else
  if ! command -v "${toolchain_gcc}" >/dev/null 2>&1; then
    echo "AArch64 toolchain not found: ${toolchain_gcc}" >&2
    echo "请安装交叉编译工具链（示例）：" >&2
    echo "  Ubuntu/Debian: sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu" >&2
    echo "  Fedora:        sudo dnf install gcc-aarch64-linux-gnu gcc-c++-aarch64-linux-gnu" >&2
    echo "或使用 --toolchain-bin 指定自带工具链的 bin 目录。" >&2
    exit 1
  fi
fi

echo "=================================================="
echo "GryFlux - AArch64 cross build"
echo "Project:   ${PROJECT_ROOT}"
echo "Build dir: ${BUILD_DIR}"
echo "Type:      ${BUILD_TYPE}"
echo "Jobs:      ${JOBS}"
echo "Toolchain: ${TOOLCHAIN_FILE}"
echo "Prefix:    ${TOOLCHAIN_PREFIX}-"
echo "Bin dir:   ${TOOLCHAIN_BIN_DIR:-<PATH>}"
echo "Sysroot:   ${SYSROOT:-<none>}"
echo "Profile:   ${DO_PROFILE}"
echo "Install:   ${DO_INSTALL}"
echo "=================================================="

if [[ "${DO_CLEAN}" == "true" ]]; then
  echo "Cleaning build dir: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

CMAKE_CONFIG_ARGS=(
  "-S" "${PROJECT_ROOT}"
  "-B" "${BUILD_DIR}"
  "-DCMAKE_TOOLCHAIN_FILE=${TOOLCHAIN_FILE}"
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  "-DBUILD_TEST=OFF"
  "-DTOOLCHAIN_PREFIX=${TOOLCHAIN_PREFIX}"
)

if [[ "${DO_PROFILE}" == "true" ]]; then
  cxx_flags="${CXXFLAGS:-} -DGRYFLUX_BUILD_PROFILING=1"
  CMAKE_CONFIG_ARGS+=("-DCMAKE_CXX_FLAGS:STRING=${cxx_flags}")
fi

if [[ -n "${TOOLCHAIN_BIN_DIR}" ]]; then
  CMAKE_CONFIG_ARGS+=("-DTOOLCHAIN_BIN_DIR=${TOOLCHAIN_BIN_DIR}")
fi

if [[ -n "${SYSROOT}" ]]; then
  CMAKE_CONFIG_ARGS+=("-DCMAKE_SYSROOT=${SYSROOT}")
fi

if [[ "${DO_INSTALL}" == "true" ]]; then
  if [[ -n "${PREFIX}" ]]; then
    CMAKE_CONFIG_ARGS+=("-DCMAKE_INSTALL_PREFIX=${PREFIX}")
  else
    CMAKE_CONFIG_ARGS+=("-DCMAKE_INSTALL_PREFIX=${PROJECT_ROOT}/install/aarch64")
  fi
fi

cmake "${CMAKE_CONFIG_ARGS[@]}" "${EXTRA_CMAKE_ARGS[@]}"
cmake --build "${BUILD_DIR}" -j "${JOBS}"

if [[ "${DO_INSTALL}" == "true" ]]; then
  cmake --install "${BUILD_DIR}"
fi

EXE="${BUILD_DIR}/src/app/example/simple_pipeline_example"
if [[ -f "${EXE}" ]]; then
  echo ""
  echo "Output:"
  echo "  ${EXE}"
  if command -v file >/dev/null 2>&1; then
    file "${EXE}" || true
  fi
fi
