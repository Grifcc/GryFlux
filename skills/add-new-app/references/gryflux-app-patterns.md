# GryFlux App 结构参考

新增部署型 app 时，优先参考这个文件。不要依赖“本地但未提交”的 app 作为唯一范本；只把它们当作补充上下文。

## 已提交模式

### `src/app/example`

- 最小框架示例
- 适合理解 `GraphTemplate`、`ResourcePool`、`AsyncPipeline`
- 不适合作为 RKNN/OpenCV 部署型 app 模板

如果仓库未来加入了已提交的部署型 app，再优先参考那些公开目录；在那之前，直接使用下面的推荐骨架。

## 代码质量约束

- 所有新增代码按 Google C++ 风格组织。
- `main` 函数只保留高层流程：
  - logger 初始化
  - 参数解析函数调用
  - 资源创建
  - 图构建
  - pipeline 运行
  - 汇总输出
- 帮助信息、参数解析、默认值整理、配置校验必须拆到独立函数或独立配置结构中。
- 借鉴其他 app 时只保留高质量模式，不要照搬差实现。发现现有实现明显混乱时，按更干净的方式重写。
- 每个 app 都必须是独立个体，不允许在源码中写死对其他 app 或外部兄弟项目的相对依赖路径。

## 开始前先确定

1. app 名称
2. 输入类型
   - 图片目录
   - 视频文件
   - 摄像头流
   - 自定义 source
3. 输出类型
   - 图片文件
   - 视频文件
   - 元数据/结果文件
   - 自定义 consumer
4. 资源类型
   - 纯 CPU
   - NPU
   - NPU 加额外状态资源
5. 第三方依赖模式
   - app-local：依赖随 app 目录一起管理
   - external-shared：依赖位于仓库外部或统一共享目录
6. 是否要加入 `src/app/CMakeLists.txt` 参与默认构建

## 推荐目录骨架

只创建需要的目录，但优先保持下面这类结构：

```text
src/app/<app-name>/
├── CMakeLists.txt
├── README.md
├── <app-name>.cpp
├── consumer/
├── context/
├── nodes/
├── packet/
└── source/
```

常见可选目录：

- `assets/`：示例素材或文档资源
- `3rdparty/`：app 自带部署依赖
- 业务算法目录，例如 `postprocess_impl/`、`tracking_core/`、`custom_ops/`

如果 `3rdparty/` 放在 app 目录里，推荐语义是：

- 目录位置固定在 `src/app/<app-name>/3rdparty/`
- 作为本地部署依赖存在，便于 app 自己管理 include/lib 路径
- 默认不提交到 git
- 需要在 app 目录或仓库根目录补 `.gitignore`
- 需要在 `README.md` 中说明依赖应如何准备

## 公开仓库可直接复用的最小示例

假设要新增 `image-classifier`：

```text
src/app/image-classifier/
├── CMakeLists.txt
├── README.md
├── image-classifier.cpp
├── consumer/
│   ├── result_writer.h
│   └── result_writer.cpp
├── context/
│   ├── npu_context.h
│   └── npu_context.cpp
├── nodes/
│   ├── classifier_nodes.h
│   ├── Input/
│   ├── Preprocess/
│   ├── Inference/
│   ├── Postprocess/
│   └── Output/
├── packet/
│   └── classifier_packet.h
└── source/
    ├── image_dir_source.h
    └── image_dir_source.cpp
```

对应 DAG 可以先从最简单的线性图开始：

```text
Input -> Preprocess -> Inference -> Postprocess -> Output
```

只有在业务确实需要时，再扩成多分支或多资源流水线。

## 主程序检查项

主程序优先模仿仓库中已提交的高质量 app 主程序；如果当前没有，就按下面这份固定清单组织：

1. 引入框架头文件：
   - `async_pipeline.h`
   - `graph_template.h`
   - `resource_pool.h`
   - `template_builder.h`
   - 如果暴露 `--profile`，再引 profiling 相关头文件
2. 引入 app 自己的头文件：
   - nodes 聚合头
   - source
   - consumer
   - context
   - 需要时引 packet
3. 设置 logger 默认行为
4. 调用独立的 CLI 参数解析函数，并在参数非法时明确报错
5. 用 `ResourcePool` 注册资源实例
6. 构建带稳定节点 ID 的 DAG
7. 构造 source 和 consumer
8. 构造并运行 `AsyncPipeline`
9. 输出吞吐或汇总统计
10. 只有在编译和运行都开启 profiling 时才输出 profiling 文件

## README 规范

部署型 app 的 `README.md` 至少保留下面六节。README 规范就在本文件这一节，从这里开始往下看。

### 1. 概述

必须写清楚：

- 这个 app 做什么
- 输入是什么
- 输出是什么
- DAG 大致结构是什么

推荐写法：

```md
## 概述

`<app-name>` 用于执行 <任务描述>。

- 输入：<输入类型>
- 输出：<输出类型>
- DAG：`Input -> Preprocess -> Inference -> Postprocess -> Output`
```

### 2. 目录说明

必须说明：

- `src/app/<app-name>/` 下关键目录各自做什么
- `3rdparty/` 只是本地部署依赖，不进 git

推荐写法：

```md
## 目录说明

- `context/`：资源上下文，例如 NPU / TensorRT 执行资源
- `source/`：输入数据读取
- `consumer/`：输出结果写出
- `packet/`：流水线数据包定义
- `nodes/`：图节点实现
- `3rdparty/`：本地部署依赖目录，仅用于本地构建和运行，不提交到 git
```

### 3. 依赖准备

必须说明：

- 需要哪些外部依赖
- 这些依赖应该放到哪里
- 如果用 `3rdparty/`，给出期望目录结构
- 模型文件是否需要单独准备

推荐写法：

```md
## 依赖准备

需要准备：

- OpenCV
- RKNN / TensorRT
- Eigen
- 模型文件

如果采用 app-local 依赖，目录期望为：

```text
src/app/<app-name>/3rdparty/
├── opencv/
├── librknn_api/
└── Eigen/
```

模型文件：

- `<model>.rknn` 或 `<model>.engine`
- 不提交到 git，运行前手动准备
```

### 4. 构建方式

必须说明：

- 宿主构建命令
- 交叉编译命令
- 需要的 CMake 变量
- 只构建本 target 的命令

推荐写法：

```md
## 构建方式

宿主构建：

```bash
cmake -S . -B build
cmake --build build --target <app-name> -j$(nproc)
```

交叉编译：

```bash
cmake -S . -B build-aarch64 \
  -DCMAKE_TOOLCHAIN_FILE=<toolchain.cmake> \
  -D<APPNAME>_3RDPARTY_ROOT=<3rdparty-root>
cmake --build build-aarch64 --target <app-name> -j$(nproc)
```

常用 CMake 变量：

- `CMAKE_TOOLCHAIN_FILE`
- `<APPNAME>_3RDPARTY_ROOT`
```

### 5. 运行方式

必须说明：

- 完整命令示例
- 必填参数
- 常用可选参数
- 输入输出示例

推荐写法：

```md
## 运行方式

```bash
./<app-name> <model_path> <input_path> <output_path> [options]
```

必填参数：

- `model_path`
- `input_path`
- `output_path`

常用可选参数：

- `--threads`
- `--max-active`
- `--profile`

输入输出示例：

- 输入：`./data/images/`
- 输出：`./outputs/`
```

### 6. 参数说明

必须说明：

- CLI 参数表
- 默认值
- 参数含义

推荐写法：

```md
## 参数说明

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--threads` | `8` | 线程池大小 |
| `--max-active` | `4` | 最大活跃 packet 数 |
| `--profile` | `false` | 是否输出 profiling 信息 |
```

### 可选：耗时与性能

只有在当前环境可以实际运行 app 并拿到有效统计时，才增加这一节。

如果只是交叉编译、静态交付，或者当前环境缺少板端/NPU/GPU/模型/数据集，README 不强制包含这一节。

可写内容：

- 总耗时的统计口径
- 吞吐的计算方式
- 建议关注的关键耗时
- 测试环境

推荐写法：

```md
## 耗时与性能

- 总耗时：从 pipeline 启动到全部结果写出完成
- 吞吐：`processed_count / total_seconds`
- 建议输出：
  - 总耗时
  - 平均每包耗时
  - 吞吐
- 测试环境：
  - 平台
  - 模型
  - 输入规模
```

## CMake 规则

### 通用规则

- 可执行文件名通常与 app 目录同名
- 增加 app 本地 include 路径：

```cmake
target_include_directories(<app> PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

- 复用 `${SRC_DIR}` 引入框架源码

### 节点和 app 本地源码收集

优先沿用仓库里的现有写法：

```cmake
file(GLOB_RECURSE APP_NODE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/nodes/*/*/*.cpp"
)

file(GLOB_RECURSE APP_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/context/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/consumer/*.cpp"
)
```

只有 app 自己确实拥有其他源码目录时再补进去。

### 第三方依赖

- 如果 app 自带部署依赖，采用 app-local 依赖管理方式：
  - 依赖目录可放在 `src/app/<app-name>/3rdparty/`
  - 实际依赖文件默认不提交到 git
  - 用 `.gitignore` 忽略 `3rdparty/` 或其中的大文件/运行库
  - 不要在代码中写 `../other-project/...` 或 `../other-app/...` 这类跨目录依赖路径
  - 定义根目录 cache 变量
  - 定义 include/lib 路径
  - 用 `if(NOT EXISTS ...) message(FATAL_ERROR ...)` 做显式检查
  - 把运行时 `.so` 安装到 `install/lib`
  - 在 `README.md` 中写清楚依赖来源、目录结构和准备步骤
- 如果 app 依赖外部统一目录，显式暴露依赖根目录变量，不要把绝对路径写死在源码里
- 如果依赖是板端专用 `aarch64` 库，不要期待宿主 `x86_64` 的全量构建一定能成功

## TensorRT 资源管理

如果新 app 使用 TensorRT，按“两层资源”管理，不要把所有对象塞进一个全局 context。

### 第一层：共享只读模型资源

这层按模型维度管理，通常全 app 只初始化一次：

- `nvinfer1::IRuntime`
- `nvinfer1::ICudaEngine`
- 反序列化后的模型与 binding 元信息
- profile / shape 约束信息

建议封装成 `TrtModelHandle`、`TrtEngineRegistry` 或同类对象，由多个执行资源共享。

### 第二层：并发执行资源

这层按并发槽位管理，并接入 `ResourcePool`：

- `nvinfer1::IExecutionContext`
- `cudaStream_t`
- device buffers
- pinned host buffers
- event 与临时执行状态

建议一个资源实例对应一个独占的 TensorRT 推理槽位。也就是说，一个 packet 在推理节点里拿到的是一个 `TrtContext`，而不是直接共享全局 execution context。

### 推荐接入方式

1. app 初始化时创建一次共享模型资源
2. 按并发数创建多个 `TrtContext`
3. 把这些 `TrtContext` 注册进 `ResourcePool`
4. 推理节点只通过 `Context` 访问 TensorRT 执行资源

推荐语义：

- `Engine` 共享
- `ExecutionContext + Stream + Buffers` 独占
- `ResourcePool` 只管理并发执行槽位，不管理模型本体

### 不建议的做法

- 每次推理时临时创建/销毁 `IExecutionContext`
- 多线程并发复用同一个 `IExecutionContext`
- 把 engine、execution context、buffer 全做成一个全局单例

### 动态 shape

- 固定 shape 模型：每个执行资源实例预分配全部 buffer
- 动态 shape 模型：优先按 profile 或输入规格分组资源
- 不要在节点执行热路径里频繁 `cudaMalloc`

## 接入规则

只有当新 app 适合进入默认构建流程时，才修改 `src/app/CMakeLists.txt`：

```cmake
add_subdirectory(<app-name>)
```

如果新 app 依赖特殊工具链或目标机运行时，也可以保持为按 target 显式构建。

## 验证清单

1. 用 app 需要的正确工具链配置 CMake
2. 先只构建新 target
3. 重点检查：
   - 头文件缺失
   - app-local include 未接好
   - 库架构不匹配
   - glibc/libstdc++ 运行时版本假设错误
4. 如果构建通过，再运行最小 smoke test
5. 把准确的构建和运行命令写进 app 的 `README.md`
