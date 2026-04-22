# GryFlux Segmentation DeepLab Ascend

## 简介

这个目录是基于 GryFlux 异步流水线框架实现的 DeepLabV3 语义分割示例，面向昇腾 NPU 推理场景。

整体流水线如下：

`source -> preprocess -> inference -> postprocess`

其中：

- `source` 负责扫描输入图片并构造 `DeepLabPacket`
- `preprocess` 负责图像预处理
- `inference` 负责调用 ACL/Ascend 模型执行推理
- `postprocess` 负责将模型输出转换为预测 mask
- `consumer` 负责将后处理结果保存到输出目录，默认最多保存 `10` 张预测图

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

## 目录说明

- `go.cpp`: 程序入口
- `context/`: 昇腾 NPU 上下文、ACL 生命周期与模型加载
- `source/`: 数据集扫描与样本生成
- `nodes/`: 输入、预处理、推理、后处理节点
- `consumer/`: 保存预测结果并输出运行摘要

当前目录中的推理上下文调用方式已经对齐 `src/app/yolox_detection_310p`：

- 入口通过 `context/infercontext.h` 中的 `CreateInferContexts(...)` 批量注册 `npu` 资源
- ACL 的 `init/setDevice/finalize` 生命周期由 `context/AclEnvironment.*` 统一托管
- `nodes/Inference/InferenceNode.cpp` 只依赖 `InferContext` 的 `copyToDevice()`、`executeModel()` 和 `copyToHost()` 接口

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
./src/app/segmentation_deeplab_ascend/deeplab <om_model_path> <input_dir> <output_dir>
```

参数说明：

- `<om_model_path>`: 模型路径，必填
- `<input_dir>`: 输入图片目录，必填
- `<output_dir>`: 输出目录，必填

程序内部会固定使用以下配置，不再需要用户传参：

- `npu_instances = 1`
- `thread_pool_size = 8`
- `max_active_packets = 4`

示例：

```bash
./src/app/segmentation_deeplab_ascend/deeplab \
  /path/to/deeplabv3_int8.om \
  /path/to/JPEGImages \
  /path/to/output_dir
```

如果想边运行边看最后一段日志：

```bash
./src/app/segmentation_deeplab_ascend/deeplab \
  /path/to/deeplabv3_int8.om \
  /path/to/JPEGImages \
  /path/to/output_dir 2>&1 | tail -n 40
```

## 数据集要求

当前实现对输入数据有几个约定：

- `image_dir` 中只会扫描扩展名为 `.jpg` 的图片

这意味着如果你的输入图片不是 `.jpg`，需要先改 `source/deeplab_source.h` 中的数据扫描逻辑。

## 输出结果

运行过程中会输出：

- 当前处理进度
- 前 `10` 张预测 mask 的保存路径
- 最终保存目录和已保存数量

预测结果会保存到你传入的 `<output_dir>`，默认最多保存 `10` 张。

## 常见注意事项

- 目录改名或迁移后，记得同步检查上层 `src/app/CMakeLists.txt` 中的 `add_subdirectory(...)`
- `om_model_path`、`input_dir` 和 `output_dir` 现在都是必填参数
- `npu_instances` 应与可用设备数量相匹配，否则 ACL 初始化或设备绑定可能失败
- 如果开启了 profiling 编译选项，运行结束后会生成 `deeplab_timeline.json`
