#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


class ImageEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_paths, input_shape, cache_file):
        super().__init__()
        self.image_paths = image_paths
        self.input_shape = tuple(int(v) for v in input_shape)
        self.cache_file = Path(cache_file)
        self.batch_size = int(self.input_shape[0])
        self.channels = int(self.input_shape[1])
        self.height = int(self.input_shape[2])
        self.width = int(self.input_shape[3])
        self.current_index = 0
        self.device_input = cuda.mem_alloc(
            trt.volume(self.input_shape) * np.dtype(np.float32).itemsize
        )

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index >= len(self.image_paths):
            return None

        batch = np.zeros(self.input_shape, dtype=np.float32)
        loaded = 0
        while loaded < self.batch_size and self.current_index < len(self.image_paths):
            image = cv2.imread(str(self.image_paths[self.current_index]), cv2.IMREAD_COLOR)
            self.current_index += 1
            if image is None or image.size == 0:
                continue

            resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            chw = np.transpose(rgb, (2, 0, 1))
            batch[loaded] = chw
            loaded += 1

        if loaded == 0:
            return None

        if loaded < self.batch_size:
            for idx in range(loaded, self.batch_size):
                batch[idx] = batch[loaded - 1]

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch))
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)


def parse_args():
    parser = argparse.ArgumentParser(description="Build TensorRT INT8 engine for ZeroDCE ONNX")
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--dataset", required=True, help="Calibration image root directory")
    parser.add_argument("--engine", required=True, help="Output TensorRT engine path")
    parser.add_argument(
        "--cache",
        default="",
        help="INT8 calibration cache path (default: next to engine with .cache suffix)",
    )
    parser.add_argument(
        "--input-shape",
        default="",
        help="Override input shape as NCHW, e.g. 1x3x640x640. Use when ONNX has dynamic dims.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=128,
        help="Maximum number of images to use for calibration",
    )
    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=2048,
        help="TensorRT workspace size in MiB",
    )
    return parser.parse_args()


def collect_images(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in exts
    )
    if not images:
        raise RuntimeError(f"No calibration images found under: {root}")
    return images


def parse_shape_text(shape_text):
    values = [int(v) for v in shape_text.lower().replace("x", " ").split()]
    if len(values) != 4:
        raise ValueError("input-shape must be provided as NCHW, for example 1x3x640x640")
    return tuple(values)


def normalize_dims(dims, override_shape):
    raw = list(int(v) for v in dims)
    if override_shape:
        return tuple(override_shape)

    normalized = []
    dynamic = False
    for idx, dim in enumerate(raw):
        if dim < 0:
            dynamic = True
            if idx == 0:
                normalized.append(1)
            else:
                normalized.append(-1)
        else:
            normalized.append(dim)

    if dynamic:
        raise RuntimeError(
            "Detected dynamic ONNX input dims. Please pass --input-shape as NCHW, e.g. 1x3x640x640"
        )

    return tuple(normalized)


def parse_onnx_network(onnx_path, input_shape_override):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        model_bytes = f.read()
    if not parser.parse(model_bytes):
        errors = []
        for idx in range(parser.num_errors):
            errors.append(str(parser.get_error(idx)))
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errors))

    if network.num_inputs != 1:
        raise RuntimeError(f"Expected 1 input tensor, got {network.num_inputs}")
    if network.num_outputs != 1:
        raise RuntimeError(f"Expected 1 output tensor, got {network.num_outputs}")

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    input_shape = normalize_dims(input_tensor.shape, input_shape_override)

    if len(input_shape) != 4:
        raise RuntimeError(f"Expected 4D NCHW input, got shape {input_shape}")
    if input_shape[0] != 1:
        raise RuntimeError(f"Current script supports batch=1 only, got input shape {input_shape}")
    if input_shape[1] != 3:
        raise RuntimeError(f"Current script supports 3-channel input only, got input shape {input_shape}")

    print("Parsed ONNX model:")
    print(f"  input : {input_tensor.name} {tuple(input_tensor.shape)} -> using {input_shape}")
    print(f"  output: {output_tensor.name} {tuple(output_tensor.shape)}")
    print(f"  layers: {network.num_layers}")

    return builder, network, input_tensor, input_shape


def build_engine(args):
    image_paths = collect_images(args.dataset)[: args.max_images]
    if len(image_paths) < 8:
        print(
            f"[WARN] Calibration images only {len(image_paths)}. INT8 quality may be unstable.",
            file=sys.stderr,
        )

    cache_path = Path(args.cache) if args.cache else Path(args.engine).with_suffix(".cache")
    input_shape_override = parse_shape_text(args.input_shape) if args.input_shape else None

    builder, network, input_tensor, input_shape = parse_onnx_network(args.onnx, input_shape_override)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)
    config.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name, input_shape, input_shape, input_shape)
    config.add_optimization_profile(profile)
    config.set_calibration_profile(profile)

    calibrator = ImageEntropyCalibrator(image_paths, input_shape, cache_path)
    config.int8_calibrator = calibrator

    print(f"Using {len(image_paths)} calibration images from {args.dataset}")
    print(f"Calibration cache: {cache_path}")
    print(f"Output engine: {args.engine}")

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT failed to build serialized INT8 engine")

    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    print(f"INT8 engine generated successfully: {engine_path}")


def main():
    args = parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
