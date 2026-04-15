#include "AtlasContext.h"
#include <iostream>

namespace {

void CheckAcl(aclError ret, const char* expr) {
    if (ret == ACL_SUCCESS) return;
    throw std::runtime_error(std::string("ACL call failed: ") + expr);
}

}  // namespace

AtlasContext::AtlasContext(int device_id, const std::string& model_path) 
    : device_id_(device_id) {
    try {
        CheckAcl(aclrtSetDevice(device_id_), "aclrtSetDevice");
        CheckAcl(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
        CheckAcl(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
        CheckAcl(aclmdlLoadFromFile(model_path.c_str(), &model_id_), "aclmdlLoadFromFile");

        model_desc_ = aclmdlCreateDesc();
        CheckAcl(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");
    } catch (...) {
        cleanup();
        throw;
    }

    std::cout << "[INFO] Device " << device_id_ << " 模型加载完成。" << std::endl;
}

AtlasContext::~AtlasContext() {
    cleanup();
}

void AtlasContext::cleanup() noexcept {
    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
}
