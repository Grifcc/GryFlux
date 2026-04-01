#pragma once
#include "framework/async_pipeline.h"
#include "acl/acl.h"
#include <iostream>
#include <vector>
#include <string>
#include <mutex>

#define ACL_CHECK(expr)                                                         \
    do {                                                                        \
        aclError ret = (expr);                                                  \
        if (ret != ACL_SUCCESS) {                                               \
            std::cerr << "ACL Error: " << ret << " at " << __FILE__ << ":"      \
                      << __LINE__ << " calling " << #expr << std::endl;         \
        }                                                                       \
    } while (0)

class AtlasContext : public GryFlux::Context {
public:
    AtlasContext(int deviceId, const std::string& modelPath) : deviceId_(deviceId) {
        ACL_CHECK(aclrtSetDevice(deviceId_));
        
        ACL_CHECK(aclrtCreateContext(&context_, deviceId_));
        ACL_CHECK(aclrtSetCurrentContext(context_)); 

        ACL_CHECK(aclmdlLoadFromFile(modelPath.c_str(), &modelId_));
        
        modelDesc_ = aclmdlCreateDesc();
        ACL_CHECK(aclmdlGetDesc(modelDesc_, modelId_));

        inputSize_ = aclmdlGetInputSizeByIndex(modelDesc_, 0);
        ACL_CHECK(aclrtMalloc(&inputDevBuffer_, inputSize_, ACL_MEM_MALLOC_HUGE_FIRST));
        inputDataset_ = aclmdlCreateDataset();
        inputDataBuffer_ = aclCreateDataBuffer(inputDevBuffer_, inputSize_);
        ACL_CHECK(aclmdlAddDatasetBuffer(inputDataset_, inputDataBuffer_));

        outputSize_ = aclmdlGetOutputSizeByIndex(modelDesc_, 0);
        ACL_CHECK(aclrtMalloc(&outputDevBuffer_, outputSize_, ACL_MEM_MALLOC_HUGE_FIRST));
        outputDataset_ = aclmdlCreateDataset();
        outputDataBuffer_ = aclCreateDataBuffer(outputDevBuffer_, outputSize_);
        ACL_CHECK(aclmdlAddDatasetBuffer(outputDataset_, outputDataBuffer_));
        
        std::cout << "[INFO] NPU " << deviceId_ << " 资源与模型加载完成。" << std::endl;
    }

    ~AtlasContext() {
        aclrtSetCurrentContext(context_);

        ACL_CHECK(aclrtFree(inputDevBuffer_));
        ACL_CHECK(aclrtFree(outputDevBuffer_));
        aclDestroyDataBuffer(inputDataBuffer_);
        aclDestroyDataBuffer(outputDataBuffer_);
        aclmdlDestroyDataset(inputDataset_);
        aclmdlDestroyDataset(outputDataset_);
        aclmdlUnload(modelId_);
        aclmdlDestroyDesc(modelDesc_);
        
        ACL_CHECK(aclrtDestroyContext(context_)); 
    }

    int getDeviceId() const { return deviceId_; }

    void executeInference(const std::vector<float>& host_input, std::vector<float>& host_output) {
        std::lock_guard<std::mutex> lock(npu_mutex_);

        ACL_CHECK(aclrtSetCurrentContext(context_));

        if (host_output.size() * sizeof(float) < outputSize_) {
            host_output.resize(outputSize_ / sizeof(float));
        }

        ACL_CHECK(aclrtMemcpy(inputDevBuffer_, inputSize_, host_input.data(), inputSize_, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(aclmdlExecute(modelId_, inputDataset_, outputDataset_));
        ACL_CHECK(aclrtMemcpy(host_output.data(), outputSize_, outputDevBuffer_, outputSize_, ACL_MEMCPY_DEVICE_TO_HOST));
    }

private:
    int deviceId_ = 0;
    aclrtContext context_ = nullptr;
    uint32_t modelId_ = 0;
    aclmdlDesc* modelDesc_ = nullptr;
    
    size_t inputSize_ = 0;
    void* inputDevBuffer_ = nullptr;
    aclmdlDataset* inputDataset_ = nullptr;
    aclDataBuffer* inputDataBuffer_ = nullptr;

    size_t outputSize_ = 0;
    void* outputDevBuffer_ = nullptr;
    aclmdlDataset* outputDataset_ = nullptr;
    aclDataBuffer* outputDataBuffer_ = nullptr;
    
    std::mutex npu_mutex_;
};