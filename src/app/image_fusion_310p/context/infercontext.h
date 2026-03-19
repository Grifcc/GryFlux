#pragma once

#include "framework/context.h"
#include "acl/acl.h"
#include <string>

// 必须继承自框架的 Context 基类
class InferContext : public GryFlux::Context {
public:
    InferContext();
    ~InferContext() override;

    // 初始化此单个 Context 实例（加载模型，分配一份独立的 Device 内存）
    bool Init(const std::string& modelPath, int deviceId);
    void Destroy();

    // --- 给 InferNode 调用的 Getter 接口 ---
    uint32_t GetModelId() const { return modelId_; }
    size_t GetInputSize() const { return modelInputSize_; }
    size_t GetOutputSize() const { return modelOutputSize_; }

    // 获取已经组装好的 Dataset，Node 拿到后直接喂给 aclmdlExecute
    aclmdlDataset* GetInputDataset() const { return inputDataset_; }
    aclmdlDataset* GetOutputDataset() const { return outputDataset_; }

    // 获取对应的 Device 内存指针，用于 Node 执行 aclrtMemcpy
    void* GetVisDevPtr() const { return visDevPtr_; }
    void* GetIrDevPtr() const { return irDevPtr_; }
    void* GetOutDevPtr() const { return outDevPtr_; }

private:
    int32_t deviceId_;
    uint32_t modelId_;
    aclmdlDesc* modelDesc_;
    
    size_t modelInputSize_;
    size_t modelOutputSize_;

    // 单个 Context 实例独享的硬件资源缓存
    aclmdlDataset* inputDataset_ = nullptr;
    aclmdlDataset* outputDataset_ = nullptr;
    void* visDevPtr_ = nullptr;
    void* irDevPtr_ = nullptr;
    void* outDevPtr_ = nullptr;
};