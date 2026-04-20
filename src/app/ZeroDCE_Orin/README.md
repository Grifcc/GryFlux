# GryFlux Framework - ZeroDCE_Orin

`ZeroDCE_Orin` 是把原先的 Atlas 示例骨架替换为 Jetson Orin + TensorRT 的最小可运行版本。

当前实现假设：

- 输入模型为 TensorRT `.engine`
- 输入输出 tensor 都是 `NCHW`
- 当前只支持 `batch=1`
- 当前只支持 `3` 通道输入输出
- 输出 tensor 空间尺寸与输入一致
- 预处理为 `BGR -> RGB -> float[0,1] -> CHW`
- 后处理会把输出 tensor 还原成图片并异步写盘

构建依赖：

- OpenCV
- CUDA
- TensorRT

运行方式：

```bash
./zero_dce_app <engine_path> <input_dir> <output_dir> [gt_dir] [--no-save] [--no-metrics] [--infer-only]
```

说明：

- 不传 `gt_dir` 时，结果表会显示 `Proxy PSNR`
- 传入 `gt_dir` 且文件名可匹配时，结果表会显示真实 `PSNR`
- 传 `--no-save` 时，不落盘输出图片
- 传 `--no-metrics` 时，不计算 `Proxy PSNR/Loss`
- 传 `--no-save --no-metrics` 时，可以排除保存和指标开销
- 传 `--infer-only` 时，会进一步跳过输出 tensor 转图片，最接近纯推理吞吐
- 运行结束会额外打印 `Preprocess / Infer / Postprocess` 三段平均耗时
- 有 `gt_dir` 时显示真实 `PSNR`，没有 `gt_dir` 时显示 `Proxy PSNR`

建议测速命令：

```bash
./zero_dce_app /workspace/zjx/model/ZeroDCE_static640_int8.engine /workspace/zjx/data/DICM /workspace/zjx/output/zero_dce_int8
./zero_dce_app /workspace/zjx/model/ZeroDCE_static640_int8.engine /workspace/zjx/data/DICM /workspace/zjx/output/zero_dce_bench --no-save --no-metrics
./zero_dce_app /workspace/zjx/model/ZeroDCE_static640_int8.engine /workspace/zjx/data/DICM /workspace/zjx/output/zero_dce_infer --infer-only
```

INT8 量化：

```bash
python3 scripts/build_int8_engine.py \
  --onnx /workspace/zjx/model/ZeroDCE_static640.onnx \
  --dataset /workspace/zjx/data/DICM \
  --engine /workspace/zjx/model/ZeroDCE_static640_int8.engine
```

一键量化并运行：

```bash
bash scripts/quantize_and_run.sh
```

常用环境变量：

- `ONNX_PATH`：默认 `/workspace/zjx/model/ZeroDCE_static640.onnx`
- `DATASET_DIR`：默认 `/workspace/zjx/data/DICM`
- `ENGINE_PATH`：默认 `3rdparty/models/ZeroDCE_static640_int8.engine`
- `CACHE_PATH`：默认 `3rdparty/models/ZeroDCE_static640_int8.cache`
- `OUTPUT_DIR`：默认 `/workspace/zjx/output/zero_dce_int8`
- `INPUT_SHAPE`：如果 ONNX 是动态输入，手动指定为 `1x3xHxW`
