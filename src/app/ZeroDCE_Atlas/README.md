# GryFlux Framework - ZeroDCE_Atlas

## 示例说明

本示例用于演示：

- 如何在 GryFlux 中搭建基于 Atlas NPU 的低照度增强异步流水线骨架
- 如何将 `DataSource -> DAG -> DataConsumer` 组织成逐帧图像处理流程
- 如何使用 `ResourcePool` 注册多卡 `AtlasContext` 并并行执行推理
- 如何将预处理、推理、后处理拆成独立节点
- 如何在主线程中等待全部任务结束并统一打印质量指标

本应用入口在 `src/app/ZeroDCE_Atlas/ZeroDCE_Atlas.cpp`，可执行文件名为 `zero_dce_app`。

需要特别说明的是：当前 `ZeroDCE_Atlas` 更偏向“Atlas 推理流水线骨架示例”。

- `AsyncDiskWriter` 组件已经接入主程序生命周期
- `PostprocessNode` 当前仍使用模拟 `PSNR / Loss / Status`
- 当前代码里还没有完整的真实图像增强后处理与落盘链路

因此，这个示例适合用来理解 GryFlux + Atlas 的流水线组织方式，而不是作为最终可直接部署的 Zero-DCE 成品。

## 快速上手

下面按 `example/README.md` 的组织方式，结合当前实现说明本示例的结构。

### 1) 定义数据包（DataPacket）

`ZeroDcePacket` 是整个流水线里的数据载体，负责保存：

- 输入图像
- 输出图像 / 输出缓冲
- ACL host/device 内存
- 帧 ID
- 后处理阶段产生的指标

当前定义位于 `packet/ZeroDce_Packet.h`：

```cpp
struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id;
    cv::Mat input_image;
    cv::Mat output_image;

    size_t data_size = 1 * 3 * 480 * 640 * sizeof(float);

    void* host_input_ptr = nullptr;
    void* host_output_ptr = nullptr;
    void* dev_input_ptr = nullptr;
    void* dev_output_ptr = nullptr;

    uint64_t getIdx() const override { return frame_id; }
};
```

当前实现里，构造函数会直接分配 ACL Host / Device 内存，析构时自动释放。

### 2) 定义数据源（DataSource）

`ZeroDceDataSource` 位于 `source/ZeroDceDataSource.cpp`，负责扫描输入目录中的图片：

```cpp
cv::glob(input_dir + "/*.jpg", image_paths_, false);
cv::glob(input_dir + "/*.png", png_paths, false);
image_paths_.insert(image_paths_.end(), png_paths.begin(), png_paths.end());
```

然后在 `produce()` 中逐帧生成 `ZeroDcePacket`：

```cpp
auto packet = std::make_unique<ZeroDcePacket>();
packet->frame_id = current_idx_;
packet->input_image = cv::imread(image_paths_[current_idx_]);
```

因此这个示例是典型的“目录扫描式流处理”。

### 3) 定义节点（NodeBase）

本示例当前包含 3 个节点：

- `PreprocessNode`
- `InferNode`
- `PostprocessNode`

#### 3.1 `PreprocessNode`

位于 `nodes/Preprocess/PreprocessNode.cpp`。  
当前实现更像占位骨架，尚未补完整的真实 Zero-DCE 输入预处理逻辑。

目前文件中主要做的是访问 `ZeroDcePacket`，并包含一个释放 `host_output_ptr` 的占位逻辑。

也就是说，如果你后续要把这个示例补成真正的 Zero-DCE 推理工程，`PreprocessNode` 是最需要继续完善的部分之一。

#### 3.2 `InferNode`

位于 `nodes/Infer/InferNode.cpp`，负责：

- 绑定 ACL Context
- 申请 device 输入输出显存
- H2D 拷贝
- 构造输入输出 dataset
- 调用 `aclmdlExecute`
- D2H 拷贝
- 释放 device 显存

核心逻辑如下：

```cpp
aclrtSetCurrentContext(atlas_ctx.GetAclContext());
aclrtMalloc((void**)&dce_packet.dev_input_ptr, dce_packet.data_size, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dce_packet.dev_input_ptr, dce_packet.data_size,
            dce_packet.input_image.data, dce_packet.data_size,
            ACL_MEMCPY_HOST_TO_DEVICE);
aclmdlExecute(atlas_ctx.GetModelId(), inputDataset, outputDataset);
```

这部分是当前 `ZeroDCE_Atlas` 中最接近真实 Atlas 推理路径的部分。

#### 3.3 `PostprocessNode`

位于 `nodes/Postprocess/PostprocessNode.cpp`。  
当前并没有做真实的 Zero-DCE 输出图像重建，而是用模拟值填充结果：

```cpp
dce_packet.image_name = "image_" + std::to_string(dce_packet.frame_id) + ".jpg";
dce_packet.int8_psnr = 58.0 + (rand() % 500) / 100.0;
dce_packet.loss = 2.0 + (rand() % 300) / 100.0;
dce_packet.status = (dce_packet.loss > 4.5) ? "⚠️" : "✅";
```

所以 README 里这里不会把它写成“已经完成真实图像增强后处理”，因为当前代码还没有到那一步。

### 4) 定义 Context（资源上下文）

`context/AtlasContext.h` 中定义了 `AtlasContext`，用于封装：

- `aclrtContext`
- `model_id`
- `model_desc`

构造阶段负责：

- 绑定设备
- 创建 ACL Context
- 加载 OM 模型
- 获取模型描述

```cpp
ret = aclrtCreateContext(&context_, device_id_);
ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);
model_desc_ = aclmdlCreateDesc();
ret = aclmdlGetDesc(model_desc_, model_id_);
```

和 `resnet_Atlas` 不同的是，当前 `ZeroDCE_Atlas` 的 `AtlasContext` 比较轻量，没有在 context 中长期持有输入输出 dataset / buffer，而是把 dataset 的构造放在了 `InferNode` 中。

### 5) 注册资源池（ResourcePool）

主程序中把两张 Atlas 设备都注册成 `atlas_npu`：

```cpp
auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
std::vector<std::shared_ptr<GryFlux::Context>> atlas_contexts;

atlas_contexts.push_back(std::make_shared<AtlasContext>(0, omModelPath));
atlas_contexts.push_back(std::make_shared<AtlasContext>(1, omModelPath));

resourcePool->registerResourceType("atlas_npu", std::move(atlas_contexts));
```

所以 `InferNode` 可以在运行时自动拿到其中一个可用 context。

### 6) 构建 DAG（GraphTemplate + TemplateBuilder）

当前计算图同样是一条很直接的链：

```cpp
auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
    [](GryFlux::TemplateBuilder *builder) {
        builder->setInputNode<PreprocessNode>("preprocess");
        builder->addTask<InferNode>("inference", "atlas_npu", {"preprocess"});
        builder->setOutputNode<PostprocessNode>("postprocess", {"inference"});
    }
);
```

节点 ID 就是：

- `preprocess`
- `inference`
- `postprocess`

### 7) 运行异步管道（AsyncPipeline）

主程序中先启动异步写盘组件，再启动流水线：

```cpp
AsyncDiskWriter::GetInstance().Start(outputDir);

GryFlux::AsyncPipeline pipeline(
    source,
    graphTemplate,
    resourcePool,
    consumer,
    kThreadPoolSize,
    kMaxActivePackets
);
```

然后主线程通过 `consumer->get_future()` 等待处理结束，最后做：

- `consumer->printMetrics()`
- `AsyncDiskWriter::GetInstance().Stop()`
- `aclFinalize()`

## 示例 DAG 结构

DAG 结构图：

![ZeroDCE Atlas DAG](assets/chart.svg)

当前模块对应关系：

- `source`：`ZeroDceDataSource`
- `packet`：`ZeroDcePacket`
- `nodes`：`PreprocessNode -> InferNode -> PostprocessNode`
- `context`：`AtlasContext`
- `consumer`：`ZeroDceResultConsumer`

## 当前实现状态

为了避免 README 把当前代码写得“比实际更完整”，这里单独说明一下实现状态：

已经具备的部分：

- 输入目录扫描
- `DataPacket` 封装
- 多卡 Atlas Context 注册
- 基于 GryFlux 的异步 DAG 调度
- ACL 推理主干路径
- 结果汇总与主线程退出同步

仍属于骨架 / 占位实现的部分：

- `PreprocessNode` 还没有补完整的真实输入预处理
- `PostprocessNode` 当前输出的是模拟指标
- `AsyncDiskWriter` 已经在主程序生命周期中启动/停止，但当前 `PostprocessNode` 还没有显式调用 `Push()`

## 资源绑定与并行关系

当前资源绑定如下：

- `CPU(绿)`：`PreprocessNode`、`PostprocessNode`
- `Atlas NPU(蓝)`：`InferNode`
- `AtlasContext(蓝)`：注册了 `Device 0` 和 `Device 1`

因此示例的并行性主要来自：

- CPU 线程池对前后处理节点的调度
- 双 Atlas Context 对推理阶段的并行承载

## 管道运行参数

当前主程序中写死的参数为：

```cpp
constexpr size_t kThreadPoolSize = 8;
constexpr size_t kMaxActivePackets = 16;
```

含义：

- `kThreadPoolSize`：CPU 节点并发度
- `kMaxActivePackets`：系统允许同时在途的 packet 数

## 构建与运行

### 1) 构建

在仓库根目录执行：

```bash
bash build.sh
```

构建完成后，可执行文件位于：

```bash
build/src/app/ZeroDCE_Atlas/zero_dce_app
```

### 2) 运行

```bash
./build/src/app/ZeroDCE_Atlas/zero_dce_app <om_model_path> <input_dir> <output_dir>
```

命令行参数共 3 个：

- `om_model_path`：Atlas OM 模型路径
- `input_dir`：输入图片目录
- `output_dir`：输出目录

当前代码支持扫描：

- `*.jpg`
- `*.png`

运行时会输出：

- 当前进度
- 每张图的模拟 `PSNR / Loss / Status`
- 平均 `PSNR / Loss`
- 总耗时与端到端 FPS

## 时间线可视化（示例资产）

当前目录下附带了一份手工整理的时间线 JSON：

- `assets/timeline_zero_dce.json`

可用于时间线网页展示：

```text
http://profile.grifcc.top:8076/
```

使用方式：

1. 浏览器打开页面
2. 选择 `assets/timeline_zero_dce.json`
3. 生成 packet 级 timeline

这份 JSON 是为了说明当前 DAG 结构和并行关系而准备的示例资产，不是程序自动导出的 profiling 结果。

## 指标说明

`ZeroDceResultConsumer` 当前会：

- 累计每个 packet 的 `image_name / int8_psnr / loss / status`
- 在最后一个 packet 到达时打印汇总表
- 统计总耗时与端到端 FPS
- 通过 promise 唤醒主线程

因此它既承担了结果消费，也承担了“结束信号”的角色。

## 退出与稳定性说明

当前实现为了保证“全部处理完成后自然退出”，做了这些安排：

- 主线程等待 `consumer->get_future()`
- pipeline 放到后台线程执行，并在结束时 `join`
- `DataSource` 在数据耗尽时正确更新 `hasMore`
- 结束阶段显式调用 `AsyncDiskWriter::Stop()`
- 最后调用 `aclFinalize()`

如果程序异常或卡住，优先检查：

- `input_dir` 中是否存在可读图片
- `output_dir` 是否可写
- 模型路径是否正确
- Ascend 运行时环境变量是否已配置
- 当前 Zero-DCE 前后处理逻辑是否已按你的模型格式补完整

## 目录结构

- `ZeroDCE_Atlas.cpp`：主程序入口
- `source/ZeroDceDataSource.h/.cpp`：输入目录扫描与 packet 生成
- `packet/ZeroDce_Packet.h`：数据包与 ACL 内存封装
- `context/AtlasContext.h/.cpp`：Atlas 上下文与模型句柄
- `nodes/Preprocess/PreprocessNode.cpp`：预处理骨架
- `nodes/Infer/InferNode.cpp`：ACL 推理执行
- `nodes/Postprocess/PostprocessNode.cpp`：后处理骨架与模拟指标
- `consumer/ResultConsumer/ZeroDceResultConsumer.h/.cpp`：结果汇总与完成信号
- `consumer/DiskWriter/AsyncDiskWriter.h/.cpp`：异步写盘组件
- `assets/chart.svg`：DAG 图
- `assets/timeline_zero_dce.json`：时间线示例数据
