#pragma once

#include "NvInfer.h"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        cudaError_t err = call;                                                                 \
        if (err != cudaSuccess) {                                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "                   \
                      << __FILE__ << ":" << __LINE__ << std::endl;                              \
            exit(-1);                                                                           \
        }                                                                                       \
    } while (0)
#endif

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

inline TRTLogger& GetTrtLogger() {
    static TRTLogger logger;
    return logger;
}

class TrtModelHandle {
public:
    TrtModelHandle(int device_id, const std::string& engine_path)
        : device_id_(device_id),
          engine_path_(engine_path) {
        CUDA_CHECK(cudaSetDevice(device_id_));

        std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open engine file: " + engine_path_);
        }

        const std::streamsize size = file.tellg();
        if (size <= 0) {
            throw std::runtime_error("Engine file is empty: " + engine_path_);
        }

        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!file.read(buffer.data(), size)) {
            throw std::runtime_error("Failed to read engine file: " + engine_path_);
        }

        runtime_.reset(nvinfer1::createInferRuntime(GetTrtLogger()));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize TensorRT engine: " + engine_path_);
        }

        ResolveTensorMetadata();
    }

    int deviceId() const { return device_id_; }

    nvinfer1::ICudaEngine* engine() const { return engine_.get(); }

    const std::string& enginePath() const { return engine_path_; }

    const std::string& inputTensorName() const { return input_tensor_name_; }

    const std::string& outputTensorName() const { return output_tensor_name_; }

    const nvinfer1::Dims4& inputDims() const { return input_dims_; }

    size_t inputSizeBytes() const { return input_size_bytes_; }

    size_t outputSizeBytes() const { return output_size_bytes_; }

private:
    struct RuntimeDeleter {
        void operator()(nvinfer1::IRuntime* runtime) const {
            delete runtime;
        }
    };

    struct EngineDeleter {
        void operator()(nvinfer1::ICudaEngine* engine) const {
            delete engine;
        }
    };

    void ResolveTensorMetadata() {
        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (tensor_name == nullptr) {
                continue;
            }

            const auto mode = engine_->getTensorIOMode(tensor_name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                input_tensor_name_ = tensor_name;
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                output_tensor_name_ = tensor_name;
            }
        }

        if (input_tensor_name_.empty() || output_tensor_name_.empty()) {
            throw std::runtime_error("Failed to resolve TensorRT input/output tensors");
        }

        // This app currently targets fixed-shape ResNet classification.
        input_dims_ = nvinfer1::Dims4{1, 3, 224, 224};
        input_size_bytes_ = static_cast<size_t>(1 * 3 * 224 * 224) * sizeof(float);
        output_size_bytes_ = static_cast<size_t>(1000) * sizeof(float);
    }

    int device_id_ = 0;
    std::string engine_path_;
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    nvinfer1::Dims4 input_dims_{1, 3, 224, 224};
    size_t input_size_bytes_ = 0;
    size_t output_size_bytes_ = 0;
};
