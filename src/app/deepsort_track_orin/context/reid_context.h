#pragma once

#include "framework/context.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class ReidContext : public GryFlux::Context {
public:
    ReidContext(
        std::shared_ptr<class SharedReidModel> shared_model,
        int device_id = 0);
    ~ReidContext() override;

    void copyToDevice(const void* data, size_t size);
    void execute();
    void copyToHost(float* output_data, size_t element_count);
    void bindCurrentThread();
    size_t getInputBufferSize() const { return input_size_; }
    size_t getOutputElementCount() const { return output_size_ / sizeof(float); }

private:
    void createExecutionContext();
    void allocateBuffers();
    void releaseBuffers();

    std::shared_ptr<SharedReidModel> shared_model_;
    int device_id_;
    void* stream_ = nullptr;
    class ExecutionContextHandle;
    std::unique_ptr<ExecutionContextHandle> execution_context_;

    std::vector<void*> bindings_;
    int input_binding_index_ = -1;
    int output_binding_index_ = -1;

    void* device_input_ptr_ = nullptr;
    void* device_output_ptr_ = nullptr;
    void* host_output_ptr_ = nullptr;
    size_t input_size_;
    size_t output_size_;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateReidInferContexts(
    const std::string& engine_path,
    int device_id,
    size_t instance_count);
