#pragma once

#include "framework/async_pipeline.h"
#include "trt_model_handle.h"

#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

class OrinContext : public GryFlux::Context {
public:
    OrinContext(int device_id, std::shared_ptr<TrtModelHandle> model_handle)
        : deviceId_(device_id),
          model_handle_(std::move(model_handle)) {
        if (!model_handle_) {
            throw std::runtime_error("OrinContext requires a valid TrtModelHandle");
        }

        CUDA_CHECK(cudaSetDevice(deviceId_));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

        context_ = model_handle_->engine()->createExecutionContext();
        if (context_ == nullptr) {
            throw std::runtime_error("Failed to create TensorRT execution context");
        }

        inputSize_ = model_handle_->inputSizeBytes();
        outputSize_ = model_handle_->outputSizeBytes();

        CUDA_CHECK(cudaHostAlloc(&hostInputPtr_, inputSize_, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&devInputPtr_, hostInputPtr_, 0));

        CUDA_CHECK(cudaHostAlloc(&hostOutputPtr_, outputSize_, cudaHostAllocMapped));
        CUDA_CHECK(cudaHostGetDevicePointer(&devOutputPtr_, hostOutputPtr_, 0));

        std::cout << "TensorRT 执行上下文已就绪, GPU " << deviceId_
                  << ", model: "
                  << model_handle_->enginePath() << std::endl;
    }

    ~OrinContext() noexcept {
        CleanupCudaResources();

        if (context_) {
            delete context_;
            context_ = nullptr;
        }
    }

    int getDeviceId() const { return deviceId_; }

    void executeInference(const std::vector<float>& host_input, std::vector<float>& host_output) {
        CUDA_CHECK(cudaSetDevice(deviceId_));

        if (host_input.size() * sizeof(float) < inputSize_) {
            throw std::runtime_error("Input buffer is smaller than TensorRT engine expects");
        }

        if (host_output.size() * sizeof(float) < outputSize_) {
            host_output.resize(outputSize_ / sizeof(float));
        }

        memcpy(hostInputPtr_, host_input.data(), inputSize_);

        const char* input_name = model_handle_->inputTensorName().c_str();
        const char* output_name = model_handle_->outputTensorName().c_str();

        if (!context_->setInputShape(input_name, model_handle_->inputDims())) {
            throw std::runtime_error("Failed to set TensorRT input shape");
        }
        context_->setTensorAddress(input_name, devInputPtr_);
        context_->setTensorAddress(output_name, devOutputPtr_);
        if (!context_->enqueueV3(stream_)) {
            throw std::runtime_error("TensorRT enqueueV3 failed");
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        memcpy(host_output.data(), hostOutputPtr_, outputSize_);
    }

private:
    static void LogCudaCleanupError(const char* operation, cudaError_t err) noexcept {
        if (err != cudaSuccess) {
            std::cerr << "CUDA cleanup warning: " << operation
                      << " failed with " << cudaGetErrorString(err) << std::endl;
        }
    }

    void CleanupCudaResources() noexcept {
        const cudaError_t set_device_err = cudaSetDevice(deviceId_);
        LogCudaCleanupError("cudaSetDevice", set_device_err);

        if (stream_ != nullptr) {
            LogCudaCleanupError("cudaStreamSynchronize", cudaStreamSynchronize(stream_));
        }

        if (hostInputPtr_ != nullptr) {
            LogCudaCleanupError("cudaFreeHost(hostInputPtr_)", cudaFreeHost(hostInputPtr_));
            hostInputPtr_ = nullptr;
            devInputPtr_ = nullptr;
        }

        if (hostOutputPtr_ != nullptr) {
            LogCudaCleanupError("cudaFreeHost(hostOutputPtr_)", cudaFreeHost(hostOutputPtr_));
            hostOutputPtr_ = nullptr;
            devOutputPtr_ = nullptr;
        }

        if (stream_ != nullptr) {
            LogCudaCleanupError("cudaStreamDestroy(stream_)", cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
    }

    int deviceId_ = 0;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<TrtModelHandle> model_handle_;
    nvinfer1::IExecutionContext* context_ = nullptr;

    size_t inputSize_ = 0;
    void* hostInputPtr_ = nullptr;
    void* devInputPtr_ = nullptr;

    size_t outputSize_ = 0;
    void* hostOutputPtr_ = nullptr;
    void* devOutputPtr_ = nullptr;
};
