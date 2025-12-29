# GryFlux Framework - Simple Pipeline Example

## 示例说明

本示例展示了如何使用 GryFlux 框架构建一个简单的数据处理流水线。

### 处理流程

```
  Input
    ├─→ ImagePreprocess(CPU) ─→ FeatExtractor(CPU) ─┐
    └─→ ObjectDetection(NPU) ───────────────────────→ ObjectTracker(Output)
```

### 运行结果分析

```
All 10 frames completed in 72 ms
Average: 7.20 ms/frame
Throughput: 138.89 fps
```

**关键观察：**
1. **多帧并行处理** - 从日志可以看到多个Frame同时在不同节点执行
2. **NPU资源调度** - 多个NPU并行工作（不同Frame使用不同NPU）
3. **事件驱动调度** - 节点完成后立即触发后继节点，无需等待
4. **高吞吐量** - 138.89 fps，远超单帧串行处理

## 编译运行

```bash
cd /workspace/GryFlux/build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j8

./src/app/example/simple_pipeline_example
```

## 代码结构

### 1. 定义资源上下文

```cpp
class SimulatedNPUContext : public GryFlux::Context {
    // 封装NPU设备ID和操作
};
```

### 2. 定义数据包

```cpp
struct SimpleDataPacket : public GryFlux::DataPacket {
    // 输入数据
    std::vector<float> rawData;

    // 预分配的中间结果
    std::vector<float> preprocessedData;
    std::vector<float> inferenceResult;
    std::vector<float> postprocessedResult;
};
```

### 3. 注册资源

```cpp
resourcePool->registerResourceType("npu", {
    std::make_shared<SimulatedNPUContext>(0),
    std::make_shared<SimulatedNPUContext>(1)
});
```

### 4. 构建图模板

```cpp
auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
    [](GryFlux::TemplateBuilder *builder)
    {
        builder->setInputNode<PipelineNodes::InputNode>("input");

        // 并行分支 1（CPU）
        builder->addTask<PipelineNodes::ImagePreprocessNode>(
            "imagePreprocess", "", {"input"});
        builder->addTask<PipelineNodes::FeatExtractorNode>(
            "featExtractor", "", {"imagePreprocess"});

        // 并行分支 2（NPU）
        builder->addTask<PipelineNodes::ObjectDetectionNode>(
            "objectDetection", "npu", {"input"});

        // 跟踪节点：跨帧依赖，全局只能有一个（抽象为资源类型 "tracker"）
        builder->addTask<PipelineNodes::ObjectTrackerNode>(
            "objectTracker", "tracker", {"objectDetection", "featExtractor"});

        // 输出节点：仅用于标记完成
        builder->setOutputNode<PipelineNodes::FinalOutputNode>(
            "output", {"objectTracker"});
    });
```

### 5. 创建处理器并提交数据

```cpp
#include "framework/async_pipeline.h"

auto source = std::make_shared<SimpleDataSource>(100);
auto consumer = std::make_shared<ResultConsumer>();

GryFlux::AsyncPipeline pipeline(
    source,
    graphTemplate,
    resourcePool,
    consumer,
    12, // 线程池大小
    16  // maxActivePackets（背压）
);

pipeline.run();
```

## 关键设计理念

### 无需多Worker线程

传统设计可能需要多个Worker线程从输入队列取数据，但在GryFlux中：

- **数据包之间的并行** 通过 ThreadPool + 事件驱动调度自动实现
- 使用 `AsyncPipeline` 时，框架会自动管理生产/消费线程并提供背压控制
- 无需为“多帧并行”专门再建一层 Worker 线程池

### 图模板复用

- 图结构只构建一次（`GraphTemplate::buildOnce()`）
- 所有数据包共享同一个图模板
- 每个数据包只需初始化执行状态（约40字节）

### 预分配中间结果

- 在 `DataPacket` 构造函数中预分配所有中间结果空间
- 运行时零malloc，避免内存碎片
- 数据局部性好，Cache命中率高

## 性能特性

### 并发度

- **Preprocess节点**：多帧并行（CPU任务）
- **Inference节点**：2个NPU并行（资源限制）
- **Postprocess节点**：多帧并行（CPU任务）

### 吞吐量

理论分析：
- 单帧处理时间：5ms(Preprocess) + 10ms(Inference) + 5ms(Postprocess) = 20ms
- 串行处理：50 fps
- 2个NPU并行 + 多核CPU：**138.89 fps**（实际测试结果）

提升：**2.78倍**

## 扩展方向

### 1. 增加资源

```cpp
// 注册更多NPU
resourcePool->registerResourceType("npu", {
    std::make_shared<SimulatedNPUContext>(0),
    std::make_shared<SimulatedNPUContext>(1),
    std::make_shared<SimulatedNPUContext>(2),  // 新增
    std::make_shared<SimulatedNPUContext>(3)   // 新增
});
```

### 2. 复杂DAG

```cpp
// 分支处理
builder->addTask<DetectNode1>("detect1", "npu", {"preprocess"});
builder->addTask<DetectNode2>("detect2", "npu", {"preprocess"});
builder->addTask<MergeNode>("merge", "", {"detect1", "detect2"});
```

### 3. 动态调整

```cpp
// 根据负载动态调整线程池大小
auto processor = std::make_shared<GryFlux::AsyncGraphProcessor>(
    graphTemplate, resourcePool,
    std::thread::hardware_concurrency()  // 使用所有核心
);
```

## 总结

GryFlux框架提供：
- ✅ 简洁的API（只需定义 Context、DataPacket 和 NodeBase 节点）
- ✅ 自动并行（无需手动管理线程）
- ✅ 资源管理（自动控制NPU/GPU等硬件并发度）
- ✅ 高性能（图模板复用、预分配、事件驱动）
- ✅ 易于调试（完整的日志输出）

适用于嵌入式AI推理、实时视频处理、传感器融合等场景。
