# yolox_detection_310p

`yolox_detection_310p` 是一个面向 Ascend 310P + ACL 的视频目标检测 deployment app。它按 GryFlux 的 app 结构组织了 `packet/context/source/consumer/nodes`，主入口只负责参数解析、ACL 初始化、资源注册、图模板构建和异步流水线启动。

## 目录

```text
src/app/yolox_detection_310p/
├── 3rdparty/
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── yolox_detection_310p.cpp
└── CMakeLists.txt
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖，不依赖其他 app 的目录。

默认约定：

- `3rdparty/opencv/`：OpenCV 安装前缀，或指向该前缀的符号链接
- `3rdparty/ascend-toolkit/latest/`：Ascend Toolkit 根目录，或指向该目录的符号链接
- `3rdparty/models/`：本地 `.om` 模型目录

如果依赖不直接放在这些目录下，可在 CMake 配置时显式传入：

```bash
cmake -S . -B build \
  -DGRYFLUX_ENABLE_YOLOX_DETECTION_310P=ON \
  -DYOLOX_310P_OPENCV_ROOT=/path/to/opencv \
  -DYOLOX_310P_ASCEND_ROOT=/path/to/ascend-toolkit/latest
```

## 构建

优先只配置并构建这个 target：

```bash
cmake -S src/app/yolox_detection_310p -B build/yolox_detection_310p
cmake --build build/yolox_detection_310p --target yolox_detection_310p -j$(nproc)
```

如果你希望把它接入仓库根构建，再使用：

```bash
cmake -S . -B build -DGRYFLUX_ENABLE_YOLOX_DETECTION_310P=ON
cmake --build build --target yolox_detection_310p -j$(nproc)
```

## 运行

```bash
./build/src/app/yolox_detection_310p/yolox_detection_310p \
  --input /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --model src/app/yolox_detection_310p/3rdparty/models/yolox.om \
  --width 640 \
  --height 640 \
  --conf 0.3 \
  --nms 0.45 \
  --threads 12 \
  --max-active 8 \
  --npu-instances 2
```

## 设计说明

- Ascend 资源通过 `ResourcePool` 以多个 `InferContext` 实例注册到 `npu` 资源类型。
- `DetectDataPacket` 会预分配输入 NCHW 缓冲和默认输出张量容量，减少热路径扩容。
- DAG 显式组织为 `input -> preprocess -> inference -> postprocess -> output`，视频读取和结果写出分别放在 `source/` 与 `consumer/`。
- `3rdparty/` 目录保留为 app 本地部署入口，默认不提交实际厂商运行库和模型文件。
- CMake 为 Ascend 常见运行库目录写入了 `RPATH`；若目标机驱动或 profiling 组件未就绪，ACL 仍可能在启动阶段输出平台相关日志。
