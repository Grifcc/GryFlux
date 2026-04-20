#pragma once

#include "framework/context.h"
#include "trt_model_handle.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <vector>

class OrinContext : public GryFlux::Context {
public:
    OrinContext(int device_id, std::shared_ptr<TrtModelHandle> model_handle);
    ~OrinContext() override;

    void executeInference(const std::vector<float>& host_input, std::vector<float>* host_output);

private:
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    std::shared_ptr<TrtModelHandle> model_handle_;
    nvinfer1::IExecutionContext* execution_context_ = nullptr;

    size_t input_size_bytes_ = 0;
    size_t output_size_bytes_ = 0;

    void* host_input_ptr_ = nullptr;
    void* device_input_ptr_ = nullptr;
    void* host_output_ptr_ = nullptr;
    void* device_output_ptr_ = nullptr;
};
