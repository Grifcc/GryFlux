#pragma once

#include "framework/context.h"

#include "acl/acl.h"
#include "acl/acl_mdl.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace resnet {

class AclInferContext : public GryFlux::Context {
public:
    struct Config {
        std::string model_path;
        int device_id = 0;
    };

    explicit AclInferContext(Config config);
    ~AclInferContext() override;

    bool init(std::string* error);

    const Config& config() const { return config_; }
    int getDeviceId() const { return device_id_; }

    size_t getNumInputs() const { return input_buffers_.size(); }
    size_t getInputBufferSize(size_t index) const { return input_buffers_[index].size; }
    size_t getInputBufferSize() const {
        return input_buffers_.empty() ? 0 : input_buffers_[0].size;
    }

    void copyToDevice(const void* host_data, size_t size);
    void copyToDevice(size_t input_index, const void* host_data, size_t size);
    void executeModel();
    void copyToHost();
    void copyToHost(size_t output_index, void* host_buffer, size_t size);

    size_t getNumOutputs() const { return output_buffers_.size(); }
    void* getOutputHostBuffer(size_t index) const {
        return output_buffers_[index].host_buffer;
    }
    size_t getOutputSize(size_t index) const { return output_buffers_[index].size; }
    aclmdlIODims getInputDims(size_t input_index) const;
    aclmdlIODims getOutputDims(size_t output_index) const;
    aclmdlIODims getCurrentOutputDims(size_t output_index) const;
    aclFormat getInputFormat(size_t input_index) const;
    aclFormat getOutputFormat(size_t output_index) const;
    aclDataType getInputDataType(size_t input_index) const;
    aclDataType getOutputDataType(size_t output_index) const;

private:
    struct ModelInput {
        void* device_buffer = nullptr;
        size_t size = 0;
    };

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

    std::vector<ModelInput> input_buffers_;
    aclmdlDataset* input_dataset_ = nullptr;

    std::vector<ModelOutput> output_buffers_;
    aclmdlDataset* output_dataset_ = nullptr;
};

std::vector<std::shared_ptr<GryFlux::Context>> CreateAclInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count);

}  // namespace resnet
