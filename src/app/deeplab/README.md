# GryFlux Deeplab

单链路 DAG：

`Input -> Preprocess -> Inference(RKNN) -> Postprocess -> Output`

## 构建

`deeplab` 默认使用 `src/app/deeplab/3rdparty` 下的依赖：

- OpenCV: `src/app/deeplab/3rdparty/opencv`
- RKNN: `src/app/deeplab/3rdparty/librknn_api`

默认 `3rdparty` 根目录可通过 CMake 覆盖：

```bash
cmake .. -DDEEPLAB_3RDPARTY_ROOT=/abs/path/to/deeplab/3rdparty
```

## 交叉编译

```bash
cd /path/to/GryFlux
mkdir -p build-aarch64
cd build-aarch64
cmake .. \
  -DCMAKE_TOOLCHAIN_FILE=../cmake/aarch64-toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DGRYFLUX_BUILD_PROFILING=1
make -j8 deeplab
```

## 运行

```bash
./deeplab <model_path> <dataset_dir> [output_dir] [--profile]
```

其中 `dataset_dir` 必须是一个图片目录，目录下直接放待分割图片，不需要子目录分层。例如：

```text
dataset_dir/
├── 0001.jpg
├── 0002.jpg
├── street.png
└── demo.bmp
```

程序当前不会递归扫描子目录，只会读取 `dataset_dir` 这一层的文件。

说明：

- 输入目录按文件名排序读取，支持 `.jpg/.jpeg/.png/.bmp`
- 每个文件会被当作一张独立图片送入分割流水线
- 如果目录里混有非图片文件，会被忽略
- 如果某张图片解码失败，该图片会被跳过，其余图片继续处理
- 预处理保持旧行为，固定 letterbox 到 `513x513`
- 后处理固定按 `21` 类语义分割解释第一个输出张量
- 输出结果会写入两个子目录（编号顺序与输入排序一致）：
  - `overlay/deeplab_%06d.jpg`：叠加可视化图
  - `mask/deeplab_%06d.png`：类别 mask（VOC 调色板彩色图）

