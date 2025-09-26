# YOLOX + DeepSORT Integration Overview

## 概述
本项目在原有 GryFlux 框架基础上新增了 `deepsort` 应用，用于实现 YOLOX 目标检测与 DeepSORT 目标跟踪的串联流程。该应用沿用 GryFlux 的流式计算框架与统一内存分配器，并结合 RKNN 推理加速器完成模型推理任务。

## 目录结构
```
src/app/deepsort/
├── CMakeLists.txt                 # 子项目构建规则
├── deepsort_stream.cpp            # 应用入口，与管线构建
├── package/                       # 数据封装类型定义
├── runtime/                       # RKNN 运行时包装、时间工具
├── tasks/
│   ├── deepsort_tracker/          # DeepSORT 跟踪任务实现
│   └── track_result_sender/       # DeepSORT 专用结果发送任务（生成带轨迹的帧并落盘）
├── sink/write_consumer/           # 结果写入消费者
└── modules/deepsort/              # DeepSORT 算法实现（含 tracker、kalman filter 等）
```

## 关键组件
- **Streaming Pipeline**：使用 `StreamingPipeline`、`PipelineBuilder`、`TaskRegistry` 实现图像读取、预处理、RKNN 推理、目标检测与 DeepSORT 跟踪的任务编排。
- **任务注册**：`deepsort_stream.cpp` 内将各任务以 `TaskRegistry` 形式注册，并直接复用 `src/app/yolox/tasks` 下的图像预处理、RKNN 推理、YOLOX 检测与结果写入实现，仅保留 `deepsort_tracker` 任务作为 DeepSORT 专属逻辑。
- **轨迹结果发送**：新增 `tasks/track_result_sender` 任务，与 YOLOX 原有结果发送器解耦，负责将 DeepSORT 轨迹信息绘制到帧上并输出到统一的 `WriteConsumer`。
- **数据生产者**：复用 `app/yolox` 中的 `ImageProducer`/`VideoProducer`，根据输入路径自动选择实现，兼容统一内存分配器。
- **DeepSORT Tracker**：内部实现卡尔曼滤波、匈牙利匹配、距离度量等模块；匈牙利求解逻辑现已直接内联在 `modules/deepsort/src/linear_assignment.cpp` 中，避免额外占位文件，继续支持类别/置信度信息传递。
- **RKNN 推理**：封装在 `tasks/rk_runner` 与 `runtime/rknn_fp.*` 中，含模型加载、输入输出映射、量化反量化流程。

## 配置要点
- **视频输出控制**：输出视频的启停、路径与帧率由 `src/app/deepsort/sink/write_consumer/write_consumer.cpp` 决定，入口 `deepsort_stream.cpp` 默认传入 `./outputs/result.mp4` 与 25 FPS，可在此处调整。
- **YOLOX 模型参数**：推理输入尺寸、NPU 绑定与量化信息由 `tasks/rk_runner/rk_runner.cpp` 统一加载；若需更换模型尺寸，应同时改动 `ImagePreprocess` 任务与 `RkRunner` 构造参数。

### YOLOX 检测（仅检测流程）
```
./install/yolox_stream assets/models/yolox.rknn assets/data/yolox_images
```
- 输入是 `assets/data/yolox_images/` 中的示例图片；运行结束后 `./outputs/output_*.jpg` 会保留标注结果。

### YOLOX + DeepSORT 追踪（检测 + 跟踪 + 视频输出）
```
./install/deepsort_stream assets/models/yolox.rknn assets/models/osnet_x0_25_market.rknn assets/data/WeChat.mp4
```
- 第一个参数为 YOLOX 检测模型，第二个参数为 DeepSORT ReID 模型，第三个参数为测试视频；
- 执行后 `./outputs/` 下会生成连续的 `frame_*.jpg` 追踪结果和 `result.mp4` 视频文件；
- 命令已于 2025-09-26 实测通过，`TrackResultSender` 平均耗时 2.48 ms，每帧都写入 `./outputs/frame_*.jpg` 与最终的 `result.mp4`（本次跑满 100 帧，平均吞吐 5.04 FPS）。

## YOLOX 单独检测验证
若需在不启用 DeepSORT 的前提下，验证 `yolox_stream` 的检测能力，可按以下流程操作：

1. 使用内置 `WeChat.mp4` 抽帧生成静态图像数据集：

	```bash
	ffmpeg -y -i assets/data/WeChat.mp4 -vf fps=1 assets/data/yolox_images/frame_%03d.jpg
	```

2. 编译并安装 YOLOX 流水线可执行文件：

	```bash
	cmake --build build --target yolox_stream -- -j1
	cmake --install build
	```

3. 运行检测程序并指定模型与图片目录：

	```bash
	cd install
	./yolox_stream ../assets/models/yolox.rknn ../assets/data/yolox_images
	```

执行完成后，日志会显示每帧 Letterbox 预处理与检测框信息，`install/outputs/output_*.jpg` 将保存带标注的检测结果图片，可用于人工复核。

## 打包说明
推荐的分发方式是保留完整源码，同时排除运行时产物 (`assets/`、`build/`、`install/`)：

1. 确认 `GryFlux/package_overview.txt` 已包含资源与重建说明。
2. 在仓库根目录运行：

	```bash
	cd /home/hzhy/userdata/userdata/jzm
	zip -r gryflux_src.zip GryFlux \
		 -x "GryFlux/assets/*" "GryFlux/build/*" "GryFlux/install/*" "GryFlux/outputs/*"
	```

3. 如需验证，执行 `zipinfo -1 gryflux_src.zip | head` 查看内容是否包含 `docs/`、`src/` 等目录。
4. 若仅需提供说明文档，可单独打包 `package_overview.txt`，但默认应提交 `gryflux_src.zip`。

解压后按照“Rebuild & Validation Checklist”重新构建即可恢复可执行文件与测试数据。

## 依赖
- OpenCV 4.5+
- Eigen3
- RKNN Runtime (`librknnrt.so`)
- GryFlux 框架自带的 `project_includes`（流式框架、日志、统一内存管理等）

## 注意事项
- DeepSORT 实现保留了原工程中的 `TRACHER_MATCHD` 结构，为兼容性增加了类型别名。
- RKNN 模型需与当前 RK 平台与 NPU 核心配置兼容。
- 若需要调整输入分辨率/阈值，可在任务注册处修改构造参数（例如预处理尺寸、检测置信度等）。
