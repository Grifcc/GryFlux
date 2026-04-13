#pragma once

#include "framework/context.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class ReidContext : public GryFlux::Context {
public:
    ReidContext(const std::string& model_path, int device_id);
    ~ReidContext() override;

    void bindCurrentThread();

    size_t getInputBufferSize() const { return input_size_; }
    size_t getOutputElementCount() const { return output_size_ / sizeof(float); }

    void copyToDevice(const void* data, size_t size);
    void execute();
    void copyToHost(float* host_output, size_t element_count);

private:
    void destroyDatasets() noexcept;
    void destroyBuffers() noexcept;

    int device_id_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;
    aclmdlDataset* input_dataset_ = nullptr;
    aclmdlDataset* output_dataset_ = nullptr;
    void* device_input_ptr_ = nullptr;
    void* device_output_ptr_ = nullptr;
    void* host_output_ptr_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateReidInferContexts(
    const std::string& model_path,
    int device_id,
    size_t instance_count);
