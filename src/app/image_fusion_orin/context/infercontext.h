#pragma once

#include "framework/context.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct FusionModelInfo {
    int model_width = 0;
    int model_height = 0;
    size_t vis_input_elements = 0;
    size_t ir_input_elements = 0;
    size_t output_elements = 0;
};

struct FusionInferResourceBundle {
    FusionModelInfo model_info;
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
};

class SharedFusionModel;

class InferContext : public GryFlux::Context {
public:
    struct TensorBuffer {
        int binding_index = -1;
        std::string name;
        nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;
        size_t byte_size = 0;
        size_t element_count = 0;
        void* device_ptr = nullptr;
        std::vector<std::uint8_t> host_buffer;
    };

    InferContext(
        std::shared_ptr<SharedFusionModel> shared_model,
        int device_id);
    ~InferContext() override;

    void bindCurrentThread();

    size_t GetInputElementCount(size_t index) const;
    size_t GetOutputElementCount() const;

    void copyInputToDevice(size_t index, const float* host_data, size_t element_count);
    void execute();
    void copyOutputToHost(float* host_data, size_t element_count);

private:
    void createExecutionContext();
    void allocateBuffers();
    void releaseBuffers();

    std::shared_ptr<SharedFusionModel> shared_model_;
    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;

    class ExecutionContextHandle;
    std::unique_ptr<ExecutionContextHandle> execution_context_;

    std::vector<TensorBuffer> input_buffers_;
    TensorBuffer output_buffer_;
    std::vector<void*> bindings_;
};

FusionInferResourceBundle CreateFusionInferResourceBundle(
    const std::string& engine_path,
    int device_id,
    size_t instance_count,
    int fallback_width,
    int fallback_height);
