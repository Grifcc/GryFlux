#include "orin_context.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {

void LogCudaCleanupError(cudaError_t err, const char* action) {
    if (err != cudaSuccess) {
        std::cerr << "[WARN] CUDA cleanup failed during " << action
                  << ": " << cudaGetErrorString(err) << std::endl;
    }
}

}  // namespace

OrinContext::OrinContext(int device_id, std::shared_ptr<TrtModelHandle> model_handle)
    : device_id_(device_id),
      model_handle_(std::move(model_handle)) {
    if (!model_handle_) {
        throw std::runtime_error("OrinContext requires a valid TrtModelHandle");
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    execution_context_ = model_handle_->engine()->createExecutionContext();
    if (execution_context_ == nullptr) {
        throw std::runtime_error("Failed to create TensorRT execution context");
    }

    input_size_bytes_ = model_handle_->inputSizeBytes();
    output_size_bytes_ = model_handle_->outputSizeBytes();

    CUDA_CHECK(cudaHostAlloc(&host_input_ptr_, input_size_bytes_, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_input_ptr_, host_input_ptr_, 0));

    CUDA_CHECK(cudaHostAlloc(&host_output_ptr_, output_size_bytes_, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&device_output_ptr_, host_output_ptr_, 0));

    if (!execution_context_->setInputShape(
            model_handle_->inputTensorName().c_str(),
            model_handle_->inputDims())) {
        throw std::runtime_error("Failed to set TensorRT input shape for ZeroDCE");
    }
    if (!execution_context_->setTensorAddress(
            model_handle_->inputTensorName().c_str(),
            device_input_ptr_)) {
        throw std::runtime_error("Failed to bind TensorRT input tensor");
    }
    if (!execution_context_->setTensorAddress(
            model_handle_->outputTensorName().c_str(),
            device_output_ptr_)) {
        throw std::runtime_error("Failed to bind TensorRT output tensor");
    }
}

OrinContext::~OrinContext() {
    LogCudaCleanupError(cudaSetDevice(device_id_), "cudaSetDevice");
    if (stream_ != nullptr) {
        LogCudaCleanupError(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    }

    if (host_input_ptr_ != nullptr) {
        LogCudaCleanupError(cudaFreeHost(host_input_ptr_), "cudaFreeHost(input)");
    }
    if (host_output_ptr_ != nullptr) {
        LogCudaCleanupError(cudaFreeHost(host_output_ptr_), "cudaFreeHost(output)");
    }
    if (stream_ != nullptr) {
        LogCudaCleanupError(cudaStreamDestroy(stream_), "cudaStreamDestroy");
    }
    if (execution_context_ != nullptr) {
        delete execution_context_;
    }
}

void OrinContext::executeInference(const std::vector<float>& host_input,
                                   std::vector<float>* host_output) {
    if (host_output == nullptr) {
        throw std::runtime_error("ZeroDCE output buffer must not be null");
    }

    if (host_input.size() * sizeof(float) != input_size_bytes_) {
        throw std::runtime_error("ZeroDCE input tensor size does not match TensorRT engine");
    }

    if (host_output->size() * sizeof(float) != output_size_bytes_) {
        host_output->assign(model_handle_->outputElementCount(), 0.0f);
    }

    CUDA_CHECK(cudaSetDevice(device_id_));
    std::memcpy(host_input_ptr_, host_input.data(), input_size_bytes_);

    if (!execution_context_->enqueueV3(stream_)) {
        throw std::runtime_error("TensorRT enqueueV3 failed for ZeroDCE");
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_));
    std::memcpy(host_output->data(), host_output_ptr_, output_size_bytes_);
}
