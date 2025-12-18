# GryFlux Framework

**高性能、事件驱动的异步 DAG 处理框架**

GryFlux 是一个专为嵌入式 AI 推理和实时数据处理设计的 C++ 框架，提供自动并行调度、硬件资源管理和流式处理能力。

---

## ✨ 核心特性

- 🚀 **真正的并行执行** - DAG 节点自动并行调度，无需手动管理线程
- 🔄 **事件驱动调度** - 节点完成立即触发后继节点，零等待
- 💎 **硬件资源管理** - 自动控制 NPU/GPU 并发度，防止资源竞争
- 🌊 **流式处理支持** - AsyncPipeline 封装 Source → Graph → Consumer 模式
- 🎯 **背压控制** - 自动限制活跃数据包数量，防止内存爆炸
- 📊 **图模板复用** - 构建一次，处理无限数据包
- ⚡ **零拷贝设计** - 智能指针管理，避免不必要的内存分配
- 🔍 **易于调试** - 完整的日志输出和状态追踪

---

## 🏗️ 架构概览

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                      AsyncPipeline                          │
│  ┌──────────┐      ┌──────────────────┐      ┌──────────┐  │
│  │  Source  │ ───> │ AsyncGraphProc.  │ ───> │ Consumer │  │
│  └──────────┘      └──────────────────┘      └──────────┘  │
│                            │                                │
│                            ▼                                │
│                    ┌──────────────┐                         │
│                    │ GraphTemplate│                         │
│                    └──────────────┘                         │
│                            │                                │
│                   ┌────────┴────────┐                       │
│                   ▼                 ▼                       │
│            ┌──────────┐      ┌─────────────┐               │
│            │  Nodes   │      │ResourcePool │               │
│            └──────────┘      └─────────────┘               │
└─────────────────────────────────────────────────────────────┘
```

#### 1. **GraphTemplate** - DAG 图模板
定义节点的执行顺序和依赖关系，支持任意复杂的 DAG 结构。

#### 2. **AsyncGraphProcessor** - 异步图处理器
基于事件驱动的调度引擎，自动并行执行 DAG 节点。

#### 3. **ResourcePool** - 硬件资源池
管理 NPU/GPU 等硬件资源，自动控制并发度。

#### 4. **AsyncPipeline** - 异步管道
封装 Source → Graph → Consumer 模式，自动管理生产者和消费者线程。

---

## 🚦 快速开始

### 示例：并行 DAG 处理

```cpp
#include "framework/async_pipeline.h"
#include "framework/template_builder.h"

// 1. 定义数据包
struct MyDataPacket : public GryFlux::DataPacket {
    int id;
    std::vector<float> rawData;
    std::vector<float> result;

    MyDataPacket()
        : id(0), rawData(256), result(256)
    {
    }
};

// 2. 定义节点
class InputNode : public GryFlux::NodeBase {
    void execute(DataPacket &packet, Context &ctx) override {
        auto &p = static_cast<MyDataPacket &>(packet);
        // rawData 已在 DataPacket 构造函数中预分配（固定大小）
        for (size_t i = 0; i < p.rawData.size(); ++i) p.rawData[i] = static_cast<float>(p.id);
    }
};

class ProcessNode : public GryFlux::NodeBase {
    void execute(DataPacket &packet, Context &ctx) override {
        auto &p = static_cast<MyDataPacket &>(packet);
        // 处理数据...
    }
};

class OutputNode : public GryFlux::NodeBase {
    void execute(DataPacket &packet, Context &ctx) override {
        (void)packet;
        (void)ctx;
    }
};

// 3. 构建 DAG
auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
    [](GryFlux::TemplateBuilder *builder) {
        builder->setInputNode<InputNode>("input");
        builder->addTask<ProcessNode>("process", "", {"input"});
        builder->setOutputNode<OutputNode>("output", {"process"});
    }
);

// 4. 创建资源池（如果需要硬件资源）
auto resourcePool = std::make_shared<GryFlux::ResourcePool>();
resourcePool->registerResourceType("npu", {
    std::make_shared<NPUContext>(0),
    std::make_shared<NPUContext>(1)
});

// 5. 创建流式管道
auto source = std::make_shared<MyDataSource>();
auto consumer = std::make_shared<MyDataConsumer>();

GryFlux::AsyncPipeline pipeline(
    source,
    graphTemplate,
    resourcePool,
    consumer,
    8  // 线程池大小
);

// 6. 运行
pipeline.run();  // 阻塞直到所有数据处理完成
```

---

## 📖 核心概念

### 1. DataPacket - 数据包

数据包是流经 DAG 的数据载体。

**设计原则**：
- ✅ **并行节点写入不同字段** - 避免数据竞争
- ✅ **预分配内存** - 避免运行时 malloc
- ✅ **继承 DataPacket 基类** - 框架统一管理

**示例**：
```cpp
struct VideoPacket : public GryFlux::DataPacket {
    int frameId;

    // 各节点的输出字段（避免并行节点冲突）
    cv::Mat rawImage;              // Input 节点
    cv::Mat preprocessedImage;     // Preprocess 节点 (并行分支1)
    std::vector<Box> detections;   // Detection 节点 (并行分支2, NPU)
    std::vector<Feature> features; // FeatureExtract 节点
    std::vector<Track> tracks;     // Tracker 节点（融合结果）
};
```

### 2. NodeBase - 节点基类

所有处理节点都继承 `NodeBase`，实现 `execute()` 方法。

**示例**：
```cpp
class ObjectDetectionNode : public GryFlux::NodeBase {
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override {
        auto &p = static_cast<VideoPacket &>(packet);
        auto &npu = static_cast<NPUContext &>(ctx);

        // NPU 操作流程
        npu.copyToDevice(p.preprocessedImage);
        npu.runInference();
        p.detections = npu.copyFromDevice();
    }
};
```

### 3. Context - 硬件资源上下文

封装硬件设备（NPU/GPU/DSP）的操作接口。

**示例**：
```cpp
class NPUContext : public GryFlux::Context {
private:
    int deviceId_;
    void *deviceMemory_;

public:
    void copyToDevice(const cv::Mat &image);
    void runInference();
    std::vector<Box> copyFromDevice();
};
```

### 4. GraphTemplate - DAG 图模板

定义节点的执行顺序和依赖关系。

**并行节点示例**：
```cpp
auto template = GryFlux::GraphTemplate::buildOnce([](auto *builder) {
    builder->setInputNode<InputNode>("input");

    // 并行分支 1
    builder->addTask<PreprocessNode>("preprocess", "", {"input"});

    // 并行分支 2 (与 preprocess 并行执行)
    builder->addTask<DetectionNode>("detection", "npu", {"input"});

    // 融合节点（等待两个分支完成）
    builder->setOutputNode<TrackerNode>("tracker", {"preprocess", "detection"});
});
```

### 5. ResourcePool - 资源池

管理硬件资源的分配和回收。

**特性**：
- ✅ 自动控制并发度（如只有 2 个 NPU，最多 2 个任务并行）
- ✅ 负载均衡（自动选择空闲资源）
- ✅ RAII 管理（自动释放资源）

**示例**：
```cpp
auto pool = std::make_shared<GryFlux::ResourcePool>();

// 注册 NPU 资源
pool->registerResourceType("npu", {
    std::make_shared<NPUContext>(0),
    std::make_shared<NPUContext>(1)
});

// 注册 GPU 资源
pool->registerResourceType("gpu", {
    std::make_shared<GPUContext>(0)
});
```

### 6. AsyncPipeline - 异步管道

封装 Source → Graph → Consumer 模式。

**自动管理**：
- ✅ 生产者线程（从 Source 获取数据）
- ✅ 消费者线程（将结果发送到 Consumer）
- ✅ 背压控制（防止内存爆炸）
- ✅ 线程生命周期（优雅启动和关闭）

**DataSource 接口**：
```cpp
class MyDataSource : public GryFlux::DataSource {
public:
    std::unique_ptr<DataPacket> produce() override {
        auto packet = std::make_unique<MyDataPacket>();
        // 填充数据...
        return packet;
    }

    bool hasMore() override {
        return !dataSource.eof();
    }
};
```

**DataConsumer 接口**：
```cpp
class MyDataConsumer : public GryFlux::DataConsumer {
public:
    void consume(std::unique_ptr<DataPacket> packet) override {
        auto &p = static_cast<MyDataPacket &>(*packet);
        // 处理结果...
    }
};
```

---

## 🎯 核心设计模式

### 1. 并行节点设计 - 避免数据竞争

❌ **错误示例**：
```cpp
struct BadPacket {
    double value;  // ❌ 多个并行节点同时写入
};

void NodeA() { packet.value *= 2; }     // 并行执行
void NodeB() { packet.value += 10; }    // 并行执行
// 结果不确定！
```

✅ **正确设计**：
```cpp
struct GoodPacket {
    double input;
    double resultA;  // NodeA 独占
    double resultB;  // NodeB 独占
    double output;   // Fusion 节点写入
};

void NodeA() { packet.resultA = packet.input * 2; }
void NodeB() { packet.resultB = packet.input + 10; }
void Fusion() { packet.output = packet.resultA + packet.resultB; }
```

### 2. 硬件资源三步法 - NPU/GPU 操作

真实的硬件操作流程：
```cpp
void HardwareNode::execute(DataPacket &packet, Context &ctx) {
    auto &npu = static_cast<NPUContext &>(ctx);

    // 步骤 1: Host → Device (DMA 传输)
    npu.copyToDevice(inputData);

    // 步骤 2: Device 计算
    npu.runCompute();

    // 步骤 3: Device → Host (DMA 传输)
    outputData = npu.copyFromDevice();
}
```

### 3. 背压控制 - 防止内存爆炸

AsyncPipeline 自动控制活跃数据包数量：
```cpp
// maxActivePackets = threadPoolSize - 1 (默认)
AsyncPipeline pipeline(source, template, pool, consumer, 8);
// 最多 7 个数据包同时在处理

// 自定义背压控制
AsyncPipeline pipeline(source, template, pool, consumer, 16, 15);
// 最多 15 个数据包同时在处理
```

---

## 📊 性能特性

### 并行调度

**自动并行度**：
- CPU 节点：多核并行（受线程池大小限制）
- NPU 节点：受硬件资源数量限制（如 2 个 NPU 最多 2 个任务并行）
- 并行分支：DAG 拓扑自动识别可并行节点

**实测性能** (示例)：
```
配置：8 线程池 + 2 个 NPU
数据包：100 个
吞吐量：182 packets/sec
平均延迟：5.5 ms/packet
```

### 内存管理

- ✅ **图模板复用** - 构建一次，处理无限数据包
- ✅ **智能指针** - unique_ptr 自动管理生命周期
- ✅ **预分配** - DataPacket 构造时预分配中间结果
- ✅ **零拷贝** - 指针传递，避免不必要的拷贝

### 调度开销

- ✅ **事件驱动** - 节点完成立即触发后继，无轮询
- ✅ **无锁队列** - ThreadSafeQueue 高效入队/出队
- ✅ **原子操作** - 依赖计数用原子操作，减少锁竞争

---

## 🛠️ 编译和安装

### 依赖

- C++17 或更高
- CMake 3.10+
- pthread

### 编译

```bash
git clone https://github.com/Grifcc/GryFlux.git
cd GryFlux
mkdir build && cd build
cmake ..
make -j8
```

### 交叉编译（AArch64）

宿主机需要安装 `aarch64-linux-gnu` 工具链（或自带工具链包的 `bin/` 目录）。

```bash
# Release 构建（输出到 build/aarch64）
./scripts/build-aarch64.sh --build-type Release

# 清理后重建
./scripts/build-aarch64.sh --clean --build-type Release

# 安装到指定目录（可用于打包/部署）
./scripts/build-aarch64.sh --install --prefix ./install/aarch64 --build-type Release
```

### 运行示例

```bash
# 并行管道示例
./src/app/example/simple_pipeline_example

# 查看详细日志（修改 simple_pipeline_example.cpp 设置 DEBUG 级别）
LOG.setLevel(GryFlux::LogLevel::DEBUG);
make -j8 && ./src/app/example/simple_pipeline_example
```

### 性能分析（Profiler）

框架内置 `GraphProfiler`，支持两类输出：
- **统计汇总**：每个节点的执行次数/平均耗时等
- **时间线（JSON）**：可视化每个数据包在各节点上的执行区间

1) 在代码中启用并导出时间线（示例）

```cpp
pipeline.setProfilingEnabled(true);
pipeline.run();
pipeline.printProfilingStats();
pipeline.dumpProfilingTimeline("graph_timeline.json");
```

2) 把 `graph_timeline.json` 转成可视化 HTML

```bash
python3 -m pip install pandas plotly
python3 scripts/plot_graph_timeline.py --input graph_timeline.json --output timeline.html
```

说明：
- 数据包很多时建议用 `--limit-packets N` 或调小 `--packets-per-page`，避免页面过重。

---

## 📚 示例程序

### 完整示例：并行目标检测 + 跟踪

**场景**：
- 两个并行分支：ImagePreprocess (CPU) 和 ObjectDetection (NPU)
- FeatExtractor 依赖 ImagePreprocess
- ObjectTracker 融合两个分支的结果

**代码**：见 `src/app/example/simple_pipeline_example.cpp`

**运行结果**：
```
========================================
  GryFlux Parallel Pipeline Example
  Demonstrates Parallel Node Execution
========================================
Graph template built with parallel nodes:
  Input -> ImagePreprocess -> FeatExtractor ┐
       └-> ObjectDetection(NPU) ─────────────→ ObjectTracker
========================================
All 100 packets completed in 548 ms
Average: 5.48 ms/packet
Throughput: 182.48 packets/sec
========================================
Verification Results:
  ✓ Success: 100 packets
  ✗ Failure: 0 packets
========================================
```

详细说明见：[src/app/example/README.md](src/app/example/README.md)

---

## 🔍 调试和日志

### 日志级别

```cpp
LOG.setLevel(GryFlux::LogLevel::DEBUG);  // 显示详细日志
LOG.setLevel(GryFlux::LogLevel::INFO);   // 只显示重要信息
LOG.setLevel(GryFlux::LogLevel::WARN);   // 只显示警告
LOG.setLevel(GryFlux::LogLevel::ERROR);  // 只显示错误
```

### 典型日志输出

**DEBUG 级别**（NPU 操作详情）：
```
[DEBUG] Packet 2: ObjectDetection starting on NPU 0 (vec size = 256)
[DEBUG] NPU 0: Copied 256 elements to device memory
[DEBUG] NPU 0: Computed on 256 elements (offset = 10.0)
[DEBUG] NPU 0: Copied 256 elements from device memory
[DEBUG] Packet 2: ObjectDetection completed on NPU 0 (result size = 256)
```

**INFO 级别**（处理进度）：
```
[INFO ] Graph template built with 5 nodes
[INFO ] AsyncGraphProcessor created with thread pool size 8, max active packets 16
[INFO ] Producer thread completed, produced 100 packets
[INFO ] Packet 0: ✓ PASS (sum = 3840.0, expected = 3840.0)
[INFO ] Consumer thread completed, consumed 100 packets
```

---

## 📐 架构设计原则

### 1. 分离关注点

- **GraphTemplate** - 定义"做什么"（DAG 拓扑）
- **NodeBase** - 定义"怎么做"（节点逻辑）
- **ResourcePool** - 管理"用什么"（硬件资源）
- **AsyncGraphProcessor** - 负责"调度"（并行执行）

### 2. 组合优于继承

- NodeBase 是最小接口（只有 execute）
- 用户通过组合 Context 获得硬件能力
- 避免深层继承树

### 3. RAII 资源管理

- 智能指针自动管理生命周期
- ResourcePool 自动分配和回收资源
- 无需手动 delete

### 4. 事件驱动

- 节点完成立即触发后继节点
- 无轮询，无忙等待
- 高效利用 CPU

---

## 🤝 贡献指南

欢迎贡献代码、报告 Bug 或提出新功能建议！

### 报告 Bug

请提供：
1. 系统环境（OS、编译器版本）
2. 复现步骤
3. 期望行为 vs 实际行为
4. 相关日志

### 提交 Pull Request

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交修改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开 Pull Request

---

## 📄 许可证

Copyright 2025 Grifcc

---

## 📧 联系方式

- GitHub: [@Grifcc](https://github.com/Grifcc)
- Issues: [https://github.com/Grifcc/GryFlux/issues](https://github.com/Grifcc/GryFlux/issues)

---

## 🙏 致谢

感谢所有贡献者和用户的支持！

---

## 📖 更多文档

- [示例程序详解](src/app/example/README.md) - 完整的并行管道示例
