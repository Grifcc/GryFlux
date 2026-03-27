#pragma once

#include "framework/context.h"
#include "acl/acl.h"
#include <string>
#include <stdexcept>
#include <iostream>

// 检查 ACL 返回值的宏
#define CHECK_ACL(ret, msg) \
    if ((ret) != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("ACL Error ") + std::to_string((int)(ret)) + ": " + (msg)); \
    }

/**
 * @brief 昇腾 NPU 硬件上下文管理器
 */
class DeepLabNPUContext : public GryFlux::Context {
public:
    DeepLabNPUContext(int device_id, const std::string& model_path) 
        : deviceId_(device_id) 
    {
        std::cout << "[NPUContext] 初始化 Device " << deviceId_ << "..." << std::endl;

        CHECK_ACL(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        CHECK_ACL(aclrtCreateContext(&context_, deviceId_), "aclrtCreateContext");
        CHECK_ACL(aclrtCreateStream(&stream_), "aclrtCreateStream");

        CHECK_ACL(aclmdlLoadFromFile(model_path.c_str(), &modelId_), "aclmdlLoadFromFile");
        modelDesc_ = aclmdlCreateDesc();
        CHECK_ACL(aclmdlGetDesc(modelDesc_, modelId_), "aclmdlGetDesc");

        inputSize_ = aclmdlGetInputSizeByIndex(modelDesc_, 0);
        outputSize_ = aclmdlGetOutputSizeByIndex(modelDesc_, 0);

        CHECK_ACL(aclrtMalloc(&dev_buf_in_, inputSize_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc dev_in");
        CHECK_ACL(aclrtMalloc(&dev_buf_out_, outputSize_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc dev_out");

        inputDataset_ = aclmdlCreateDataset();
        aclDataBuffer* in_buf = aclCreateDataBuffer(dev_buf_in_, inputSize_);
        CHECK_ACL(aclmdlAddDatasetBuffer(inputDataset_, in_buf), "Add input buf");

        outputDataset_ = aclmdlCreateDataset();
        aclDataBuffer* out_buf = aclCreateDataBuffer(dev_buf_out_, outputSize_);
        CHECK_ACL(aclmdlAddDatasetBuffer(outputDataset_, out_buf), "Add output buf");
    }

    void bindToCurrentThread()
    {
        CHECK_ACL(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    }

    ~DeepLabNPUContext() override {
        std::cout << "[NPUContext] 清理 Device " << deviceId_ << " 资源..." << std::endl;
        aclrtSetCurrentContext(context_);

        if (inputDataset_) aclmdlDestroyDataset(inputDataset_);
        if (outputDataset_) aclmdlDestroyDataset(outputDataset_);
        if (dev_buf_in_) aclrtFree(dev_buf_in_);
        if (dev_buf_out_) aclrtFree(dev_buf_out_);
        if (modelDesc_) aclmdlDestroyDesc(modelDesc_);
        if (modelId_) aclmdlUnload(modelId_);
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);
        
        aclrtResetDevice(deviceId_);
    }

    // --- Getters 提供给 InferenceNode 使用 ---
    aclrtStream getStream() const { return stream_; }
    uint32_t getModelId() const { return modelId_; }
    aclmdlDataset* getInputDataset() const { return inputDataset_; }
    aclmdlDataset* getOutputDataset() const { return outputDataset_; }
    void* getDevBufIn() const { return dev_buf_in_; }
    void* getDevBufOut() const { return dev_buf_out_; }
    size_t getInputSize() const { return inputSize_; }
    size_t getOutputSize() const { return outputSize_; }

private:
    int deviceId_;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t modelId_ = 0;
    aclmdlDesc* modelDesc_ = nullptr;
    size_t inputSize_ = 0;
    size_t outputSize_ = 0;
    void* dev_buf_in_ = nullptr;
    void* dev_buf_out_ = nullptr;
    aclmdlDataset* inputDataset_ = nullptr;
    aclmdlDataset* outputDataset_ = nullptr;
};
