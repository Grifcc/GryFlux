#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash build.sh [--profile]

Options:
  --profile   Build with graph profiling compiled in (-DGRYFLUX_BUILD_PROFILING=1)
EOF
}

profile=0
while [[ $# -gt 0 ]]; do
    case "$1" in
    --profile)
        profile=1
        shift
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

cmake .. \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_INSTALL_PREFIX=../install \
    -DBUILD_TEST=False \
    -DCMAKE_CXX_FLAGS:STRING="${CXXFLAGS:-} ${extra_cxxflags}"

make install -j"$(nproc)"

cd ..
