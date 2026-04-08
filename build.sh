#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash build.sh [--enable_profile] [--toolchain <path>]

Options:
  --enable_profile   Build with graph profiling compiled in (-DGRYFLUX_BUILD_PROFILING=1)
  --toolchain PATH   CMake toolchain file for cross compilation
EOF
}

profile=0
toolchain_file=""
while [[ $# -gt 0 ]]; do
    case "$1" in
    --enable_profile)
        profile=1
        shift
        ;;
    --toolchain)
        if [[ $# -lt 2 ]]; then
            echo "Missing value for --toolchain"
            usage
            exit 2
        fi
        toolchain_file="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown arg: $1"
        usage
        exit 2
        ;;
    esac
done

extra_cxxflags=""
if [[ "$profile" -eq 1 ]]; then
    extra_cxxflags="-DGRYFLUX_BUILD_PROFILING=1"
fi

mkdir -p build
cd build

cmake_args=(
    ..
    -DCMAKE_BUILD_TYPE="Release"
    -DCMAKE_INSTALL_PREFIX=../install
    -DBUILD_TEST=False
    -DCMAKE_CXX_FLAGS:STRING="${CXXFLAGS:-} ${extra_cxxflags}"
)

if [[ -n "$toolchain_file" ]]; then
    cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}")
fi

cmake "${cmake_args[@]}" \

make install -j"$(nproc)"

cd ..
