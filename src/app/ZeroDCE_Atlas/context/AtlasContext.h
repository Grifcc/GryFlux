#pragma once
#include <string>
#include "framework/context.h"
#include "acl/acl.h"

class AtlasContext : public GryFlux::Context {
public:
    AtlasContext(int device_id, const std::string& model_path);
    
    ~AtlasContext();

    aclrtContext GetAclContext() const { return context_; }
    uint32_t GetModelId() const { return model_id_; }
    aclmdlDesc* GetModelDesc() const { return model_desc_; }

private:
    int device_id_;
    aclrtContext context_ = nullptr;
    uint32_t model_id_ = 0;
    aclmdlDesc* model_desc_ = nullptr;
};