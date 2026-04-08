#pragma once

#include "framework/context.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

struct ModelOutput {
    void* device_buffer = nullptr;
    void* host_buffer = nullptr;   
    size_t size = 0;
};

class InferContext : public GryFlux::Context {
public:
    InferContext(const std::string& engine_model_path, int device_id = 0);
    ~InferContext() override;

    void bindCurrentThread();

    size_t getInputBufferSize() const { return input_buffer_size_; }

    void copyToDevice(const void* host_data, size_t size);
    void executeModel();
    void copyToHost();

    size_t getNumOutputs() const { return output_buffers_.size(); }
    void* getOutputHostBuffer(size_t index) const { return output_buffers_[index].host_buffer; }
    size_t getOutputSize(size_t index) const { return output_buffers_[index].size; }

private:
    void loadEngine(const std::string& engine_model_path);
    void allocateBuffers();
    void releaseBuffers();

    template <typename T>
    struct TrtDeleter {
        void operator()(T* ptr) const;
    };

    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;

    void* input_buffer_ = nullptr;
    size_t input_buffer_size_ = 0;
    int input_index_ = -1;

    std::vector<ModelOutput> output_buffers_;
    std::vector<int> output_indices_;
    std::vector<void*> bindings_;
};
