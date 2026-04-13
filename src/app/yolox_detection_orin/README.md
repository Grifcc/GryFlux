# yolox_detection_orin

`yolox_detection_orin` 是一个面向 Orin + TensorRT 的视频目标检测 deployment app。它按 GryFlux 的 app 结构组织了 `packet/context/source/consumer/nodes`，主入口只负责参数解析、资源注册、图模板构建和异步流水线启动。

## 目录

```text
src/app/yolox_detection_orin/
├── 3rdparty/
├── app_options.h/.cpp
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── yolox_detection_orin.cpp
└── CMakeLists.txt
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖，不依赖其他 app 的目录。

默认约定：

- `3rdparty/opencv/`：OpenCV 安装前缀，或指向该前缀的符号链接
- `3rdparty/cuda/`：CUDA Toolkit 根目录，或指向该目录的符号链接
- `3rdparty/TensorRT/`：TensorRT 根目录，或指向该目录的符号链接
- `3rdparty/models/`：本地模型文件目录

如果你的依赖不直接放在这些目录下，可以在 CMake 配置时显式传入：

```bash
cmake -S . -B build \
  -DYOLOX_ORIN_OPENCV_ROOT=/path/to/opencv \
  -DYOLOX_ORIN_CUDA_ROOT=/path/to/cuda \
  -DYOLOX_ORIN_TENSORRT_ROOT=/path/to/TensorRT
```

## 构建

优先只构建这个 target：

```bash
cmake --build build --target yolox_detection_orin -j$(nproc)
```

## 运行

```bash
./build/src/app/yolox_detection_orin/yolox_detection_orin \
  --input /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --model src/app/yolox_detection_orin/3rdparty/models/yolox.engine \
  --width 640 \
  --height 640 \
  --conf 0.3 \
  --nms 0.45 \
  --threads 12 \
  --max-active 8 \
  --npu-instances 2
```

## 设计说明

- TensorRT 资源按“共享模型 + 独占执行槽位”接入：engine 在 app 内共享，execution context、stream 和 buffers 为每个 `InferContext` 独占。
- `DetectDataPacket` 会预分配输入 NCHW 缓冲和默认输出张量容量，减少热路径扩容。
- DAG 显式组织为 `input -> preprocess -> inference -> postprocess -> output`，视频读取和结果写出分别放在 `source/` 与 `consumer/`。
