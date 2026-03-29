#pragma once
#include <string>
#include "framework/context.h"
#include "acl/acl.h"
// #include "framework/context.h" // 如果 GryFlux 有基础 Context 类，取消注释并继承

class AtlasContext : public GryFlux::Context {
public:
    // 构造函数：负责绑定设备并加载模型
    AtlasContext(int device_id, const std::string& model_path);
    
    // 析构函数：负责安全释放显存和模型描述
    ~AtlasContext();

    // 提供给 InferNode 调用的接口
    aclrtContext GetAclContext() const { return context_; }
    uint32_t GetModelId() const { return model_id_; }
    aclmdlDesc* GetModelDesc() const { return model_desc_; }

private:
    int device_id_;
    aclrtContext context_ = nullptr;
    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;
};