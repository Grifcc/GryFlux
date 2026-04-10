# GryFlux ResNet

## 概述

`resnet` 用于执行基于 RKNN 的图像分类流水线。

- 输入：图片目录
- 输出：分类可视化图片和每张图片的 Top-K 文本结果
- DAG：`Input -> Preprocess -> Inference -> Postprocess -> Output`

## 目录说明

- `context/`：ResNet 的 RKNN NPU 资源上下文
- `source/`：图片目录读取器
- `consumer/`：分类结果写出器
- `packet/`：流水线数据包定义
- `nodes/`：Input / Preprocess / Inference / Postprocess / Output 节点实现
- `3rdparty/`：本地部署依赖目录，用于本地构建和运行

## 依赖准备

`resnet` 默认使用 `src/app/resnet/3rdparty` 下的依赖：

- OpenCV: `src/app/resnet/3rdparty/opencv`
- RKNN: `src/app/resnet/3rdparty/librknn_api`

## 构建方式

交叉编译：

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-toolchain.cmake \
  -DGRYFLUX_BUILD_PROFILING=1 \
  -DRESNET_3RDPARTY_ROOT=/abs/path/to/resnet/3rdparty
cmake --build build-aarch64 --target resnet -j$(nproc)
```

## 运行方式

```bash
./resnet <model_path> <dataset_dir> <synset_path> [output_dir] [options] [--profile]
```


输入目录只扫描当前层级，支持 `.jpg/.jpeg/.png/.bmp`，按文件名排序。

## 输出说明

输出目录会包含两个子目录：

- `images/`：带 Top-K 分类结果的可视化图片
- `labels/`：每张图片对应的文本结果，格式为 `rank class_id probability label`
