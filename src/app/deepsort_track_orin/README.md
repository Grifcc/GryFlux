# deepsort_track_orin

`deepsort_track_orin` 是一个面向 Orin + TensorRT 的多目标跟踪 deployment app。它按 GryFlux app 结构组织了 `packet/context/source/consumer/nodes`，并把检测模型与 ReID 模型都接到了 `ResourcePool` 中。

## 目录

```text
src/app/deepsort_track_orin/
├── 3rdparty/
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── utils/
├── deepsort_track_orin.cpp
└── CMakeLists.txt
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖。

- `3rdparty/opencv/`
- `3rdparty/cuda/`
- `3rdparty/TensorRT/`
- `3rdparty/Eigen/`
- `3rdparty/models/`

如果依赖目录不在这些默认位置，可在配置时显式传入：

```bash
cmake -S . -B build \
  -DDEEPSORT_ORIN_OPENCV_ROOT=/path/to/opencv \
  -DDEEPSORT_ORIN_CUDA_ROOT=/path/to/cuda \
  -DDEEPSORT_ORIN_TENSORRT_ROOT=/path/to/TensorRT \
  -DDEEPSORT_ORIN_EIGEN_ROOT=/path/to/eigen
```

## 构建

```bash
cmake --build build --target deepsort_track_orin -j$(nproc)
```

## 运行

```bash
./build/src/app/deepsort_track_orin/deepsort_track_orin \
  --input /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --yolox src/app/deepsort_track_orin/3rdparty/models/yolox.engine \
  --reid src/app/deepsort_track_orin/3rdparty/models/reid.engine
```

## 设计说明

- DAG 结构为 `input -> preprocess -> detection_inference -> postprocess -> reid_preprocess -> reid_inference -> output`。
- 检测模型和 ReID 模型都按“共享模型 + 独占执行槽位”方式注册到 `ResourcePool`。
- `TrackDataPacket` 预分配检测输入 NCHW 缓冲、YOLOX 默认输出张量容量，以及跟踪阶段常用容器容量。
- DeepSORT 跟踪器保留在 `consumer/` 阶段串行执行，保证输出视频顺序稳定。
