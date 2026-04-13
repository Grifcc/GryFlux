#pragma once

#include "framework/context.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class SharedDetectionModel;

class InferContext : public GryFlux::Context {
public:
    InferContext(
        std::shared_ptr<SharedDetectionModel> shared_model,
        int device_id = 0);
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
    struct ModelOutput {
        void* device_buffer = nullptr;
        void* host_buffer = nullptr;
        size_t size = 0;
    };

    void createExecutionContext();
    void allocateBuffers();
    void releaseBuffers();

    std::shared_ptr<SharedDetectionModel> shared_model_;
    int device_id_ = 0;
    void* stream_ = nullptr;
    class ExecutionContextHandle;
    std::unique_ptr<ExecutionContextHandle> execution_context_;

    void* input_buffer_ = nullptr;
    size_t input_buffer_size_ = 0;
    int input_index_ = -1;

    std::vector<ModelOutput> output_buffers_;
    std::vector<int> output_indices_;
    std::vector<void*> bindings_;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateDetectionInferContexts(
    const std::string& engine_model_path,
    int device_id,
    size_t instance_count);
