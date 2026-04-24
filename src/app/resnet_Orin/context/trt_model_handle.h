#pragma once

#include "NvInfer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
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
            throw std::runtime_error(                                                            \
                std::string("CUDA Error: ") + cudaGetErrorString(err) +                         \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));                           \
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

    static size_t TensorDataTypeSize(nvinfer1::DataType data_type) {
        switch (data_type) {
            case nvinfer1::DataType::kFLOAT:
                return sizeof(float);
            case nvinfer1::DataType::kHALF:
                return sizeof(uint16_t);
            case nvinfer1::DataType::kINT8:
            case nvinfer1::DataType::kUINT8:
            case nvinfer1::DataType::kFP8:
                return sizeof(uint8_t);
            case nvinfer1::DataType::kINT32:
                return sizeof(int32_t);
            case nvinfer1::DataType::kINT64:
                return sizeof(int64_t);
            case nvinfer1::DataType::kBOOL:
                return sizeof(bool);
            default:
                throw std::runtime_error("Unsupported TensorRT tensor data type");
        }
    }

    static size_t TensorVolume(const nvinfer1::Dims& dims, const std::string& tensor_name) {
        if (dims.nbDims <= 0) {
            throw std::runtime_error("Tensor " + tensor_name + " has invalid TensorRT shape");
        }

        size_t volume = 1;
        for (int i = 0; i < dims.nbDims; ++i) {
            if (dims.d[i] <= 0) {
                throw std::runtime_error(
                    "Tensor " + tensor_name + " uses dynamic or non-positive dimensions");
            }
            volume *= static_cast<size_t>(dims.d[i]);
        }
        return volume;
    }

    void RequireFloatTensor(const std::string& tensor_name) const {
        if (engine_->getTensorDataType(tensor_name.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "resnet_Orin expects float TensorRT tensors for host-side preprocessing/postprocessing");
        }
    }

    size_t TensorSizeBytes(const std::string& tensor_name) const {
        const nvinfer1::Dims dims = engine_->getTensorShape(tensor_name.c_str());
        return TensorVolume(dims, tensor_name) *
               TensorDataTypeSize(engine_->getTensorDataType(tensor_name.c_str()));
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
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
                ++output_count;
                output_tensor_name_ = tensor_name;
            }
        }

        if (input_count != 1 || output_count != 1) {
            throw std::runtime_error(
                "resnet_Orin expects exactly one TensorRT input tensor and one output tensor");
        }

        const nvinfer1::Dims input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
        if (input_dims.nbDims != 4) {
            throw std::runtime_error("resnet_Orin expects a 4D TensorRT input tensor");
        }
        if (input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
            input_dims.d[2] != 224 || input_dims.d[3] != 224) {
            throw std::runtime_error(
                "resnet_Orin expects TensorRT input shape [1, 3, 224, 224]");
        }

        RequireFloatTensor(input_tensor_name_);
        RequireFloatTensor(output_tensor_name_);

        input_dims_ = nvinfer1::Dims4{
            input_dims.d[0],
            input_dims.d[1],
            input_dims.d[2],
            input_dims.d[3],
        };
        input_size_bytes_ = TensorSizeBytes(input_tensor_name_);
        output_size_bytes_ = TensorSizeBytes(output_tensor_name_);
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
