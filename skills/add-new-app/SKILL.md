---
name: add-new-app
description: 在 GryFlux 仓库中新增一个模型部署 app，包括 `src/app/<app-name>` 目录布局、packet/context/source/consumer/node 脚手架、主程序流水线接线、CMake 接入和构建验证。用于用户要求新增或迁移一个推理应用，新增模型部署流水线，补齐 RKNN/OpenCV/Eigen 等依赖接线，或让新 app 可以在当前仓库中构建和提交。
---

# 新增模型 App

## 概览

优先按当前仓库中已提交的公开结构创建新应用；如果仓库里还没有可参考的部署型 app，也按本 skill 给出的推荐骨架直接落地。除非确实必要，否则不要顺手改动 GryFlux 核心框架。

## 工作流

1. 开始前先阅读 [references/gryflux-app-patterns.md](./references/gryflux-app-patterns.md)。
2. 先判断仓库中是否存在已提交的相似 app：
   - 如果有，优先在其目录结构和主程序接线方式上做同构扩展
   - 如果没有，就直接使用参考文档中的推荐骨架
   - `src/app/example` 只适合框架演示，不适合作为部署型 app 模板
3. 在 `src/app/<app-name>/` 下创建新目录，只创建当前 app 真正需要的子目录。
4. 先实现 app 自己的类型，再写主程序：
   - `packet/`
   - `context/`
   - `source/`
   - `consumer/`
   - `nodes/`
5. 在类型齐备后再实现 `<app-name>.cpp`，接好以下内容：
   - logger 初始化
   - CLI 参数解析
   - `ResourcePool` 资源注册
   - `GraphTemplate::buildOnce(...)`
   - source / consumer 构造
   - `AsyncPipeline`
6. 新增或更新 `src/app/<app-name>/CMakeLists.txt`。
7. 判断是否要把新目录加入 `src/app/CMakeLists.txt`：
   - 只有当这个 app 适合进入默认构建流程时才加 `add_subdirectory(...)`
8. 优先只构建新 target 做验证，不要一上来全量构建整个仓库。

## 示例

用户请求：

```text
新增一个 image-classifier app，输入是图片目录，输出是每张图片的分类结果文本文件，模型走 RKNN。
```

执行时按下面的最小方案落地：

1. 创建 `src/app/image-classifier/`
2. 至少补齐这些文件：
   - `image-classifier.cpp`
   - `CMakeLists.txt`
   - `README.md`
   - `packet/classifier_packet.h`
   - `context/npu_context.h/.cpp`
   - `source/image_dir_source.h/.cpp`
   - `consumer/result_writer.h/.cpp`
   - `nodes/Input/`
   - `nodes/Preprocess/`
   - `nodes/Inference/`
   - `nodes/Postprocess/`
   - `nodes/Output/`
3. 在主程序里完成：
   - `ResourcePool` 注册 `npu`
   - `GraphTemplate::buildOnce(...)` 构建 `input -> preprocess -> inference -> postprocess -> output`
   - source/consumer 创建
   - `AsyncPipeline` 运行
4. 为该 app 单独添加 target，并优先执行：

```bash
cmake --build build --target image-classifier -j$(nproc)
```

如果当前环境无法运行，至少把构建链路、依赖路径和 README 补齐。

## 约束

- 除非用户明确要求，否则保持 GryFlux 现有框架接口不变。
- 优先复用仓库已有规范，而不是抽象出一层新的通用框架。
- 如果 app 自带 RKNN/OpenCV/Eigen 等部署依赖，优先采用 app-local 管理方式。
- 如果 app 的 3rdparty 依赖放在 `src/app/<app-name>/3rdparty/`，默认将其视为本地部署资源：目录保留在 app 下，但不提交到 git；同时补齐 `.gitignore`、路径变量和 README 获取说明。
- 如果 app 使用 TensorRT，按“共享模型资源 + 独占执行资源”的两层方式接入 `ResourcePool`，不要把 engine、execution context、stream、buffer 全部做成一个全局单例。
- 做验证时优先构建最小 target，尤其是仓库内不同 app 可能依赖不同工具链或目标机运行时。
- 交叉编译时默认假设宿主机和目标机架构不同，不要用放宽链接检查的方式“修复”问题。

## 交付物

除非用户明确缩小范围，否则应交付：

- `src/app/<app-name>/` 新目录
- 可构建的新 app target
- 带 CLI 和 pipeline 接线的主入口
- 必要的 packet/context/source/consumer/node 头文件与实现
- 对部署型 app，补一份符合本 skill 规范的 `README.md`
- 如果 app 使用 app-local 3rdparty，补 `.gitignore` 和依赖准备说明，但不要把实际大文件或厂商运行库提交进仓库
- 如果后端为 TensorRT，明确区分共享只读模型对象和并发执行槽位

## 验证

优先使用目标级构建命令，例如：

```bash
cmake --build build --target <app-name> -j$(nproc)
```

如果当前环境能运行，就做最小 smoke test。不能运行时，明确说明阻塞点属于：
- 工具链
- 架构不匹配
- 运行时依赖
- 数据集或模型文件缺失
