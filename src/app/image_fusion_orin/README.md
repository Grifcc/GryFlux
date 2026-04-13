# image_fusion_orin

`image_fusion_orin` 是一个面向 Orin + TensorRT 的图像融合 deployment app。它按 GryFlux app 结构拆分为 `packet/context/source/consumer/nodes`，并把 TensorRT 推理资源通过 `ResourcePool` 以“共享模型 + 独占执行槽位”的方式接入。

## 目录

```text
src/app/image_fusion_orin/
├── 3rdparty/
├── consumer/
├── context/
├── nodes/
├── packet/
├── source/
├── image_fusion_orin.cpp
├── CMakeLists.txt
└── README.md
```

## 依赖接入

该 app 只从自己的 `3rdparty/` 入口目录或显式 CMake 变量接入依赖。

- `3rdparty/opencv/`
- `3rdparty/cuda/`
- `3rdparty/TensorRT/`
- `3rdparty/models/`

如依赖不在默认位置，可显式传入：

```bash
cmake -S . -B build \
  -DIMAGE_FUSION_ORIN_OPENCV_ROOT=/path/to/opencv \
  -DIMAGE_FUSION_ORIN_CUDA_ROOT=/path/to/cuda \
  -DIMAGE_FUSION_ORIN_TENSORRT_ROOT=/path/to/TensorRT
```

`3rdparty/` 是本地部署资源目录，默认不提交到 git。请自行放置或软链依赖和模型文件。

## 构建

```bash
cmake --build build --target image_fusion_orin -j$(nproc)
```

## 运行

```bash
./build/src/app/image_fusion_orin/image_fusion_orin \
  --vis /path/to/vis_dir \
  --ir /path/to/ir_dir \
  --output /path/to/output_dir \
  --model /path/to/fusion.engine
```

可选参数：

- `--threads <num>`: AsyncPipeline 线程池大小，默认 `8`
- `--packets <num>`: 最大活跃 packet 数量，默认 `16`
- `--npu-instances <num>`: TensorRT 执行槽位数量，默认 `2`
- `--device <num>`: CUDA device id，默认 `0`
- `--width <num>`: 动态模型 fallback 输入宽度，默认 `640`
- `--height <num>`: 动态模型 fallback 输入高度，默认 `480`

## 设计说明

- DAG 结构为 `input -> preprocess -> inference -> postprocess -> output`
- `FusionDataPacket` 在构造阶段预分配融合流程用到的 OpenCV Mat 缓冲
- `ResultConsumer` 在 consumer 端按 packet 序号重排后顺序落盘
- TensorRT 资源按“共享只读 engine + 独占 execution context/stream/buffer”注册到 `ResourcePool`
