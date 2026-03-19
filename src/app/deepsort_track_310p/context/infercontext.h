#pragma once

#include "framework/context.h"
#include "acl/acl.h"
#include "acl/acl_mdl.h"
#include <string>
#include <vector>

struct ModelOutput {
    void* device_buffer = nullptr;
    void* host_buffer = nullptr;   
    size_t size = 0;
};

class InferContext : public GryFlux::Context {
public:
    /**
     * @brief 
     */
    InferContext(const std::string& om_model_path, int device_id = 0);
    
    /**
     * @brief
     */
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
    int device_id_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    
    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;
    
    void* input_buffer_ = nullptr;
    size_t input_buffer_size_ = 0;
    aclmdlDataset* input_dataset_ = nullptr;

    std::vector<ModelOutput> output_buffers_;
    aclmdlDataset* output_dataset_ = nullptr;
};