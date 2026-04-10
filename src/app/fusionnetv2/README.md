# GryFlux FusionNetV2

单链路 DAG：

`Input -> Preprocess -> Inference(RKNN) -> Compose -> Output`

## 构建

`fusionnetv2` 默认使用 `src/app/fusionnetv2/3rdparty` 下的依赖：

- OpenCV: `src/app/fusionnetv2/3rdparty/opencv`
- RKNN: `src/app/fusionnetv2/3rdparty/librknn_api`

默认 `3rdparty` 根目录可通过 CMake 覆盖：

```bash
cmake .. -DFUSIONNETV2_3RDPARTY_ROOT=/abs/path/to/fusionnetv2/3rdparty
```

## 交叉编译

```bash
cd /path/to/root
mkdir -p build-aarch64
cd build-aarch64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRYFLUX_BUILD_PROFILING=1 \
make -j8 fusionnetv2
```

如需覆盖默认依赖目录，可继续追加：

```bash
-DFUSIONNETV2_3RDPARTY_ROOT=/abs/path/to/fusionnetv2/3rdparty
```

## 运行

```bash
./fusionnetv2 <model_path> <dataset_root> [output_dir] [--profile]
```

说明：

- 数据集根目录要求包含 `visible/` 与 `infrared/` 子目录。
- 输入配对按文件名匹配并排序，支持 `.jpg/.jpeg/.png/.bmp`。
- 预处理保持旧行为，自动 resize 到 `640x480`。
- 输出文件按输入可见光图同名写入输出目录。
