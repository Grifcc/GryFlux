# resnet_Orin

## 概述

`resnet_Orin` 用于在 GryFlux 中执行基于 TensorRT 的 ResNet 图像分类评估。

- 输入：TensorRT `.engine` 文件、图片目录、ground truth 标签文件
- 输出：终端中的进度、总耗时、吞吐量、Top-1、Top-5
- DAG：`Preprocess -> Inference -> Postprocess`

主程序入口是 [resnet_Orin.cpp](/workspace/zjx/GryFlux/src/app/resnet_Orin/resnet_Orin.cpp)，生成的可执行文件名为 `resnet_Orin`。

## 目录说明

- `context/`：TensorRT 资源封装，包括共享模型句柄 `TrtModelHandle` 和执行槽位 `OrinContext`
- `source/`：输入数据源，当前为 `ResNetDataSource`
- `consumer/`：结果汇总与输出，当前为 `ResNetResultConsumer`
- `packet/`：流水线数据包定义，当前为 `ResNetPacket`
- `nodes/`：图节点实现，包括 `PreprocessNode`、`ResNetInferNode`、`PostprocessNode`
- `assets/`：文档配图和 timeline 示例资源
- `3rdparty/`：该 app 的本地依赖入口目录，不提交实际大文件
- `scripts/`：本地依赖准备脚本

## 依赖准备

运行当前 app 需要准备：

- OpenCV
- CUDA Runtime
- TensorRT
- ResNet TensorRT engine 文件
- 图片目录
- ground truth 标签文件

当前 app 预留了 app-local 依赖入口：

```text
src/app/resnet_Orin/3rdparty/
├── README.md
├── opencv/
├── cuda/
├── tensorrt/
└── models/
```

其中 `opencv/`、`cuda/`、`tensorrt/`、`models/` 可以是实际目录，也可以是软链接，但入口目录固定为 `src/app/resnet_Orin/3rdparty/`。

模型文件需要手动准备，例如：

- `/workspace/zjx/model/resnet50_int8.engine`

当前工作区中默认模型入口为：

- `/workspace/zjx/GryFlux/src/app/resnet_Orin/3rdparty/models/resnet50_int8.engine`

数据集和标签文件示例：

- 图片目录：`/workspace/zjx/data/imagenette2-320/val`
- 标签文件：`/workspace/zjx/data/imagenette_gt.txt`

标签文件格式为：

```text
<relative_image_path> <label_index>
```

程序启动时会先校验 `dataset_dir` 与 `gt_file_path` 是否完全匹配；不匹配时直接退出。

如果需要在当前机器上补齐本地依赖入口，可以运行：

```bash
bash /workspace/zjx/GryFlux/src/app/resnet_Orin/scripts/setup_local_3rdparty_links.sh
```

## 构建方式

宿主构建：

```bash
cmake -S /workspace/zjx/GryFlux -B /workspace/zjx/GryFlux/build
cmake --build /workspace/zjx/GryFlux/build --target resnet_Orin -j$(nproc)
```

只构建当前 target：

```bash
cmake --build /workspace/zjx/GryFlux/build --target resnet_Orin -j$(nproc)
```

如需强制从 app-local `3rdparty/` 解析依赖，可使用：

```bash
cmake -S /workspace/zjx/GryFlux -B /workspace/zjx/GryFlux/build \
  -DRESNET_ORIN_USE_LOCAL_3RDPARTY=ON \
  -DRESNET_ORIN_3RDPARTY_ROOT=/workspace/zjx/GryFlux/src/app/resnet_Orin/3rdparty
cmake --build /workspace/zjx/GryFlux/build --target resnet_Orin -j$(nproc)
```

如果只想安装当前 app，可以使用组件安装：

```bash
cmake --install /workspace/zjx/GryFlux/build \
  --prefix /workspace/zjx/GryFlux/install \
  --component resnet_Orin
```

## 运行方式

命令格式：

```bash
./resnet_Orin <engine_path> <dataset_dir> <gt_file_path>
```

推荐命令：

```bash
cd /workspace/zjx/GryFlux/build
/workspace/zjx/GryFlux/build/src/app/resnet_Orin/resnet_Orin \
  /workspace/zjx/GryFlux/src/app/resnet_Orin/3rdparty/models/resnet50_int8.engine \
  /workspace/zjx/data/imagenette2-320/val \
  /workspace/zjx/data/imagenette_gt.txt
```

输出示例：

```text
[INFO] 已处理 3900 / 3925 张图片...

========================================
[INFO] 所有 3925 张图片已处理完成。
总耗时: 7.07208 秒
吞吐量 (FPS): 555 帧/秒
有效图片数: 3925
跳过图片数: 0
Top-1 准确率: 83.2611%
Top-5 准确率: 97.5541%
========================================
```

当前 app 不会逐张写结果文件；终端会输出处理进度和最终汇总信息。

## 参数说明

| 参数 | 含义 |
| --- | --- |
| `engine_path` | TensorRT engine 文件路径 |
| `dataset_dir` | 图片目录路径，目录内容需与 GT 完全匹配 |
| `gt_file_path` | ground truth 标签文件路径 |

当前代码中的固定运行参数：

| 配置项 | 当前值 | 含义 |
| --- | --- | --- |
| `kThreadPoolSize` | `8` | 线程池大小 |
| `kMaxActivePackets` | `16` | 最大活跃 packet 数 |
| `kOrinContextInstances` | `2` | TensorRT 执行上下文实例数 |

当前实现采用两层 TensorRT 资源管理：

- 共享层：`TrtModelHandle` 持有 `IRuntime`、`ICudaEngine` 和张量元信息
- 执行层：两个 `OrinContext` 实例各自持有独占的 `IExecutionContext`、CUDA stream 和输入输出缓冲

补充说明：

- `ResNetPacket` 在构造函数中预分配了输入和输出 buffer
- `PreprocessNode` 采用短边缩放到 `256`、中心裁剪到 `224x224`、ImageNet `mean/std` 归一化
- `assets/chart.svg` 是当前 DAG 结构示意图
- `assets/timeline_resnet.json` 是 timeline 示例资产
