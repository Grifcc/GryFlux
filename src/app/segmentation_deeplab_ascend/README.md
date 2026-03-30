# GryFlux Segmentation DeepLab Ascend

## 简介

这个目录是基于 GryFlux 异步流水线框架实现的 DeepLabV3 语义分割示例，面向昇腾 NPU 推理场景。

整体流水线如下：

`source -> preprocess -> inference -> postprocess -> miou`

其中：

- `source` 负责扫描输入图片并构造 `DeepLabPacket`
- `preprocess` 负责图像预处理
- `inference` 负责调用 ACL/Ascend 模型执行推理
- `postprocess` 负责将模型输出转换为预测 mask
- `gt_process` 和 `miou` 负责读取标注并统计每张图与整套数据集的 MIoU

## 模型介绍

当前示例使用的是 DeepLabV3 语义分割模型的 Ascend OM 版本，用于对输入图像做像素级分类。

- 任务类型: 语义分割
- 输入规格: `3 x 513 x 513`
- 输出规格: `65 x 65 x 21`
- 类别数量: `21` 类, 与 VOC 数据集类别定义一致

当前代码中的模型处理流程如下：

- 前处理会将输入图片 resize 到 `513 x 513`
- 图像从 BGR 转为 RGB
- 像素值按 `x / 127.5 - 1.0` 归一化到 `[-1, 1]`
- 模型输出会在每个像素位置上对 `21` 个类别做 argmax
- 后处理会将预测结果用最近邻插值 resize 回原图尺寸

评估阶段会结合标签图统计每张图的 `image MIoU`，并汇总输出整套数据集的 `Dataset MIoU`。

## 目录说明

- `go.cpp`: 程序入口
- `context/`: 昇腾 NPU 上下文与模型加载
- `source/`: 数据集扫描与样本生成
- `nodes/`: 预处理、推理、后处理、指标计算节点
- `consumer/`: 汇总输出每张图和数据集的 MIoU

## 依赖

运行和编译该示例前，需要准备以下依赖：

- OpenCV
- Ascend CANN ACL 运行时与头文件
- CMake
- 支持 C++17 的编译器

当前 `CMakeLists.txt` 会优先从下面这些位置查找 ACL：

- `/usr/local/Ascend/ascend-toolkit/latest`
- `/usr/local/Ascend/nnrt/latest`

如果 ACL 没有安装在这些默认目录，需要同步修改当前目录下的 `CMakeLists.txt`。

## 构建

在项目根目录执行：

```bash
cmake -S /path/to/GryFlux -B /path/to/GryFlux/build
cmake --build /path/to/GryFlux/build --target deeplab -j
```

构建成功后，二进制通常位于：

```bash
/path/to/GryFlux/build/src/app/segmentation_deeplab_ascend/deeplab
```

如果执行了安装步骤：

```bash
cmake --install /path/to/GryFlux/build
```

则默认会安装到：

```bash
/path/to/GryFlux/install/bin/deeplab
```

如果你显式设置了 `CMAKE_INSTALL_PREFIX`，则以你设置的前缀为准。

## 运行

程序参数如下：

```bash
./src/app/segmentation_deeplab_ascend/deeplab <image_dir> <label_dir> <model_path> [npu_instances] [thread_pool_size] [max_active_packets]
```

参数说明：

- `<image_dir>`: 输入图片目录，必填
- `<label_dir>`: 标签目录，必填
- `<model_path>`: 模型路径，必填
- `[npu_instances]`: NPU 实例数，可选，默认 `1`
- `[thread_pool_size]`: 线程池大小，可选，默认 `8`
- `[max_active_packets]`: 最大并发包数，可选，默认 `4`

示例：

```bash
./src/app/segmentation_deeplab_ascend/deeplab \
  /path/to/JPEGImages \
  /path/to/SegmentationClass \
  /path/to/deeplabv3_int8.om \
  1 8 4
```

如果想边运行边看最后一段日志：

```bash
./src/app/segmentation_deeplab_ascend/deeplab \
  /path/to/JPEGImages \
  /path/to/SegmentationClass \
  /path/to/deeplabv3_int8.om \
  1 8 4 2>&1 | tail -n 40
```

## 数据集要求

当前实现对输入数据有几个约定：

- `image_dir` 中只会扫描扩展名为 `.jpg` 的图片
- `label_dir` 中的标签文件会按图片文件名 stem 自动匹配为同名 `.png`
- 例如 `2007_000032.jpg` 对应的标签应为 `2007_000032.png`

这意味着如果你的输入图片不是 `.jpg`，或者标签命名规则不同，需要先改 `source/deeplab_source.h` 中的数据扫描逻辑。

## 输出结果

运行过程中会输出：

- 每张图片的 `image MIoU`
- 当前处理进度
- 每个类别的 IoU
- 整体数据集的 `Dataset MIoU`
- 平均单图 `Average packet MIoU`

类别名当前按 VOC 21 类定义，统计逻辑位于 `consumer/deeplab_result_consumer.cpp`。

## 常见注意事项

- 目录改名或迁移后，记得同步检查上层 `src/app/CMakeLists.txt` 中的 `add_subdirectory(...)`
- `model_path` 现在是必填参数，运行时必须显式传入 om 模型路径
- 当前程序会校验 `image_dir`、`label_dir` 和 `model_path` 是否存在，不存在会直接抛错退出
- `npu_instances` 应与可用设备数量相匹配，否则 ACL 初始化或设备绑定可能失败
- 如果开启了 profiling 编译选项，运行结束后会生成 `deeplab_timeline.json`
