#pragma once

#include "framework/context.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct FusionModelInfo {
    int model_width = 0;
    int model_height = 0;
    size_t input_element_count = 0;
    size_t output_element_count = 0;
};

struct FusionInferResourceBundle {
    FusionModelInfo model_info;
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
};

class InferContext : public GryFlux::Context {
public:
    InferContext(const std::string& model_path, int device_id);
    ~InferContext() override;

    void bindCurrentThread();

    size_t GetInputElementCount(size_t index) const;
    size_t GetOutputElementCount() const;

    void copyInputToDevice(size_t index, const float* host_data, size_t element_count);
    void execute();
    void copyOutputToHost(float* host_data, size_t element_count);

private:
    struct TensorBuffer {
        void* device_ptr = nullptr;
        void* host_ptr = nullptr;
        size_t byte_size = 0;
        size_t element_count = 0;
    };

    void destroyDatasets() noexcept;
    void destroyBuffers() noexcept;

    int device_id_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;
    aclmdlDataset* input_dataset_ = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;
    std::vector<TensorBuffer> input_buffers_;
    TensorBuffer output_buffer_;
};

FusionInferResourceBundle CreateFusionInferResourceBundle(
    const std::string& model_path,
    int device_id,
    size_t instance_count,
    int fallback_width,
    int fallback_height);
