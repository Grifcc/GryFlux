# GryFlux RealESRGAN

单链路 DAG：

`Input -> Preprocess -> Inference(RKNN) -> Postprocess -> Output`

## 构建

`realesrgan` 支持两种依赖模式，不是只能用 EasyStream：

1. 默认模式（推荐）：`REALESRGAN_USE_EASYSTREAM_3RDPARTY=ON`  
与 `tracker` 完全一致，复用 EasyStream 的 OpenCV/RKNN/Eigen。

2. 系统库模式：`REALESRGAN_USE_EASYSTREAM_3RDPARTY=OFF`  
使用系统 OpenCV + 手动指定 RKNN 头/库路径。

### 模式 A：统一到 tracker（默认）

统一交叉编译示例：

```bash
cd /workspace/gxh/GryFlux
mkdir -p build-aarch64
cd build-aarch64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/linaro-7.5-aarch64-linux-gnu.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRYFLUX_BUILD_PROFILING=1
make -j8 realesrgan tracker
```

默认 `EASYSTREAM_ROOT` 为 `../EasyStream`，可覆盖：

```bash
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/linaro-7.5-aarch64-linux-gnu.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRYFLUX_BUILD_PROFILING=1 \
  -DEASYSTREAM_ROOT=/path/to/EasyStream
```

### 模式 B：系统库模式（可选）

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DREALESRGAN_USE_EASYSTREAM_3RDPARTY=OFF \
  -DRKNN_INCLUDE_DIR=/path/to/librknn_api/include \
  -DRKNN_LIBRARY=/usr/lib/librknnrt.so \
  -DOpenCV_DIR=/path/to/opencv/cmake
```

系统库模式下，`RKNN_INCLUDE_DIR` 必填。

## 运行

```bash
./realesrgan <model_path> <dataset_dir> [output_dir] \
  [--npu-instances N] [--threads N] [--max-active N] [--profile]
```

说明：

- 输入目录仅处理 `.jpg/.jpeg/.png`，按文件名排序。
- 预处理为固定尺寸校验（默认 256x256），不自动 resize。
- 输出图按输入同名写入输出目录。


