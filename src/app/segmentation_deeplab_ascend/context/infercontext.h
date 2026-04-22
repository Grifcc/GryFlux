#pragma once

#include "framework/context.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class InferContext : public GryFlux::Context {
public:
    struct Config {
        std::string model_path;
        int device_id = 0;
    };

    explicit InferContext(Config config);
    ~InferContext() override;

    bool init(std::string* error);

    const Config& config() const { return config_; }

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

    void destroyDatasets() noexcept;
    void destroyBuffers() noexcept;
    void unloadModel() noexcept;

    Config config_;
    bool initialized_ = false;
    int device_id_ = 0;

    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;

    void* input_buffer_ = nullptr;
    size_t input_buffer_size_ = 0;
    aclmdlDataset* input_dataset_ = nullptr;

    std::vector<ModelOutput> output_buffers_;
    aclmdlDataset* output_dataset_ = nullptr;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count);
