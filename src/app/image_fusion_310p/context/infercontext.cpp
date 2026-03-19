#include "infercontext.h"
#include <iostream>

#define CHECK_ACL(x) do { \
    aclError __ret = x; \
    if (__ret != ACL_ERROR_NONE) { \
        std::cerr << __FILE__ << ":" << __LINE__ << " ACL Error: " << __ret << std::endl; \
        return false; \
    } \
} while(0)

InferContext::InferContext() 
    : deviceId_(0), modelId_(0), modelDesc_(nullptr), 
      modelInputSize_(0), modelOutputSize_(0) {}

InferContext::~InferContext() {
    Destroy();
}

bool InferContext::Init(const std::string& modelPath, int deviceId) {
    deviceId_ = deviceId;
    CHECK_ACL(aclrtSetDevice(deviceId_));

    // 1. 加载模型与获取尺寸信息
    CHECK_ACL(aclmdlLoadFromFile(modelPath.c_str(), &modelId_));
    modelDesc_ = aclmdlCreateDesc();
    CHECK_ACL(aclmdlGetDesc(modelDesc_, modelId_));
    
    modelInputSize_ = aclmdlGetInputSizeByIndex(modelDesc_, 0); 
    modelOutputSize_ = aclmdlGetOutputSizeByIndex(modelDesc_, 0);

    // ========================================================
    // 2. 预先分配并组装此 Context 实例专属的数据集和内存
    // ========================================================
    
    // --- 准备 Input ---
    inputDataset_ = aclmdlCreateDataset();
    
    // Vis 通道
    CHECK_ACL(aclrtMalloc(&visDevPtr_, modelInputSize_, ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* visBuffer = aclCreateDataBuffer(visDevPtr_, modelInputSize_);
    CHECK_ACL(aclmdlAddDatasetBuffer(inputDataset_, visBuffer));

    // Ir 通道
    CHECK_ACL(aclrtMalloc(&irDevPtr_, modelInputSize_, ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* irBuffer = aclCreateDataBuffer(irDevPtr_, modelInputSize_);
    CHECK_ACL(aclmdlAddDatasetBuffer(inputDataset_, irBuffer));

    // --- 准备 Output ---
    outputDataset_ = aclmdlCreateDataset();
    CHECK_ACL(aclrtMalloc(&outDevPtr_, modelOutputSize_, ACL_MEM_MALLOC_NORMAL_ONLY));
    aclDataBuffer* outBuffer = aclCreateDataBuffer(outDevPtr_, modelOutputSize_);
    CHECK_ACL(aclmdlAddDatasetBuffer(outputDataset_, outBuffer));

    return true;
}

void InferContext::Destroy() {
    // 销毁 Dataset 结构
    if (inputDataset_) {
        aclmdlDestroyDataset(inputDataset_);
        inputDataset_ = nullptr;
    }
    if (outputDataset_) {
        aclmdlDestroyDataset(outputDataset_);
        outputDataset_ = nullptr;
    }

    // 释放 Device 内存
    if (visDevPtr_) { aclrtFree(visDevPtr_); visDevPtr_ = nullptr; }
    if (irDevPtr_)  { aclrtFree(irDevPtr_); irDevPtr_ = nullptr; }
    if (outDevPtr_) { aclrtFree(outDevPtr_); outDevPtr_ = nullptr; }

    // 释放模型资源
    if (modelDesc_) { aclmdlDestroyDesc(modelDesc_); modelDesc_ = nullptr; }
    if (modelId_ != 0) { aclmdlUnload(modelId_); modelId_ = 0; }
}