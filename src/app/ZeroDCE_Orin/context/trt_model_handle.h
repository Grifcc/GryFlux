#pragma once

#include "NvInfer.h"
#include "../packet/zero_dce_packet.h"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
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
            throw std::runtime_error(cudaGetErrorString(err));                                  \
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
        ValidateZeroDceShape();
    }

    int deviceId() const { return device_id_; }
    const std::string& enginePath() const { return engine_path_; }
    nvinfer1::ICudaEngine* engine() const { return engine_.get(); }
    const std::string& inputTensorName() const { return input_tensor_name_; }
    const std::string& outputTensorName() const { return output_tensor_name_; }
    const nvinfer1::Dims& inputDims() const { return input_dims_; }
    const nvinfer1::Dims& outputDims() const { return output_dims_; }
    int inputBatchSize() const { return input_dims_.d[0]; }
    int inputChannels() const { return input_dims_.d[1]; }
    int inputHeight() const { return input_dims_.d[2]; }
    int inputWidth() const { return input_dims_.d[3]; }
    int outputBatchSize() const { return output_dims_.d[0]; }
    int outputChannels() const { return output_dims_.d[1]; }
    int outputHeight() const { return output_dims_.d[2]; }
    int outputWidth() const { return output_dims_.d[3]; }
    size_t inputElementCount() const { return input_element_count_; }
    size_t outputElementCount() const { return output_element_count_; }
    size_t inputSizeBytes() const { return input_element_count_ * sizeof(float); }
    size_t outputSizeBytes() const { return output_element_count_ * sizeof(float); }

private:
    struct RuntimeDeleter {
        void operator()(nvinfer1::IRuntime* runtime) const { delete runtime; }
    };

    struct EngineDeleter {
        void operator()(nvinfer1::ICudaEngine* engine) const { delete engine; }
    };

    static nvinfer1::Dims NormalizeDims(nvinfer1::Dims dims) {
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] == -1 && i == 0) {
                dims.d[i] = 1;
                continue;
            }
            if (dims.d[i] < 0) {
                throw std::runtime_error("ZeroDCE_Orin currently supports only fixed-shape TensorRT engines");
            }
        }
        return dims;
    }

    static size_t ElementCount(const nvinfer1::Dims& dims) {
        return std::accumulate(
            dims.d,
            dims.d + dims.nbDims,
            static_cast<size_t>(1),
            [](size_t acc, int dim) { return acc * static_cast<size_t>(dim); });
    }

    void RequireFloatTensor(const std::string& tensor_name) const {
        if (engine_->getTensorDataType(tensor_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "ZeroDCE_Orin expects FP32 TensorRT input/output tensors");
        }
    }

    void ResolveTensorMetadata() {
        int input_count = 0;
        int output_count = 0;

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* tensor_name = engine_->getIOTensorName(i);
            if (tensor_name == nullptr) {
                continue;
            }

            const auto mode = engine_->getTensorIOMode(tensor_name);
            if (mode == nvinfer1::TensorIOMode::kINPUT) {
                ++input_count;
                input_tensor_name_ = tensor_name;
                input_dims_ = NormalizeDims(engine_->getTensorShape(tensor_name));
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                ++output_count;
                output_tensor_name_ = tensor_name;
                output_dims_ = NormalizeDims(engine_->getTensorShape(tensor_name));
            }
        }

        if (input_count != 1 || output_count != 1) {
            throw std::runtime_error(
                "ZeroDCE_Orin expects exactly one TensorRT input tensor and one output tensor");
        }

        RequireFloatTensor(input_tensor_name_);
        RequireFloatTensor(output_tensor_name_);

        input_element_count_ = ElementCount(input_dims_);
        output_element_count_ = ElementCount(output_dims_);
    }

    void ValidateZeroDceShape() const {
        if (input_dims_.nbDims != 4 || output_dims_.nbDims != 4) {
            throw std::runtime_error(
                "ZeroDCE_Orin expects 4D TensorRT tensors shaped as NCHW");
        }

        if (inputBatchSize() != 1 || outputBatchSize() != 1) {
            throw std::runtime_error(
                "ZeroDCE_Orin currently supports only batch size 1");
        }

        if (inputChannels() != 3 || outputChannels() != 3) {
            throw std::runtime_error(
                "ZeroDCE_Orin currently supports only 3-channel input/output tensors");
        }

        if (outputHeight() != inputHeight() || outputWidth() != inputWidth()) {
            throw std::runtime_error(
                "ZeroDCE_Orin currently expects output H/W to match input H/W");
        }
    }

    int device_id_ = 0;
    std::string engine_path_;
    std::unique_ptr<nvinfer1::IRuntime, RuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, EngineDeleter> engine_;
    std::string input_tensor_name_;
    std::string output_tensor_name_;
    nvinfer1::Dims input_dims_{};
    nvinfer1::Dims output_dims_{};
    size_t input_element_count_ = 0;
    size_t output_element_count_ = 0;
};
