# deepsort_track_310p

`deepsort_track_310p` 是一个面向 Ascend 310P + ACL 的多目标跟踪 deployment app。它按 GryFlux app 结构组织了 `packet/context/source/consumer/nodes`，并把检测模型与 ReID 模型都接到了 `ResourcePool` 中。

## 目录

```text
src/app/deepsort_track_310p/
├── 3rdparty/
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── utils/
├── deepsort_track_310p.cpp
└── CMakeLists.txt
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖。

- `3rdparty/opencv/`
- `3rdparty/ascend-toolkit/latest/`
- `3rdparty/Eigen/`
- `3rdparty/models/`

如果依赖目录不在这些默认位置，可在配置时显式传入：

```bash
cmake -S src/app/deepsort_track_310p -B build/deepsort_track_310p \
  -DDEEPSORT_310P_OPENCV_ROOT=/path/to/opencv \
  -DDEEPSORT_310P_ASCEND_ROOT=/path/to/ascend-toolkit/latest \
  -DDEEPSORT_310P_EIGEN_ROOT=/path/to/eigen
```

## 构建

优先独立构建这个 app：

```bash
cmake -S src/app/deepsort_track_310p -B build/deepsort_track_310p
cmake --build build/deepsort_track_310p --target deepsort_track_310p -j$(nproc)
```

## 运行

```bash
./build/deepsort_track_310p/deepsort_track_310p \
  --input /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --yolox src/app/deepsort_track_310p/3rdparty/models/yolox.om \
  --reid src/app/deepsort_track_310p/3rdparty/models/reid.om
```

## 设计说明

- DAG 结构为 `input -> preprocess -> detection_inference -> postprocess -> reid_preprocess -> reid_inference -> output`。
- 检测模型和 ReID 模型都通过 ACL `Context` 多实例注册到 `ResourcePool`。
- `TrackDataPacket` 预分配检测输入 NCHW 缓冲、YOLOX 默认输出张量容量，以及跟踪阶段常用容器容量。
- DeepSORT 跟踪器保留在 `consumer/` 阶段串行执行，保证输出视频顺序稳定。
- CMake 为 Ascend 常见运行库目录写入了 `RPATH`；若目标机驱动或 profiling 组件未就绪，ACL 仍可能在启动阶段输出平台相关日志。
