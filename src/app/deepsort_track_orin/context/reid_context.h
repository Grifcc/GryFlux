#pragma once

#include "framework/context.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

class ReidContext : public GryFlux::Context {
public:
    ReidContext(const std::string& engine_path, int device_id = 0);
    ~ReidContext() override;

    void copyToDevice(const void* data, size_t size);
    void execute();
    std::vector<float> copyToHost(int feature_dim = 512);
    void bindCurrentThread();

private:
    void loadEngine(const std::string& engine_path);
    void allocateBuffers();
    void releaseBuffers();

    template <typename T>
    struct TrtDeleter {
        void operator()(T* ptr) const;
    };

    int device_id_;
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;

    std::vector<void*> bindings_;
    int input_binding_index_ = -1;
    int output_binding_index_ = -1;

    void* device_input_ptr_ = nullptr;
    void* device_output_ptr_ = nullptr;
    void* host_output_ptr_ = nullptr;
    size_t input_size_;
    size_t output_size_;
};
