# GryFlux RealESRGAN

单链路 DAG：

`Input -> Preprocess -> Inference(RKNN) -> Postprocess -> Output`

## 构建

`realesrgan` 固定使用 `src/app/realesrgan/3rdparty` 下的依赖：

- OpenCV: `src/app/realesrgan/3rdparty/opencv`
- RKNN: `src/app/realesrgan/3rdparty/librknn_api`

默认 `3rdparty` 根目录为 `src/app/realesrgan/3rdparty`，也可覆盖：

```bash
cmake .. -DREALESRGAN_3RDPARTY_ROOT=/abs/path/to/realesrgan/3rdparty
```

交叉编译示例：

```bash
cd /path/to/root
mkdir -p build-aarch64
cd build-aarch64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/linaro-7.5-aarch64-linux-gnu.toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRYFLUX_BUILD_PROFILING=1 \
make -j8 realesrgan
```


## 运行

```bash
./realesrgan <model_path> <dataset_dir> [output_dir] \
  [--profile]
```

说明：

- 输入目录仅处理 `.jpg/.jpeg/.png`，按文件名排序。
- 预处理为固定尺寸校验（默认 256x256），不自动 resize。
- 输出图按输入同名写入输出目录。
