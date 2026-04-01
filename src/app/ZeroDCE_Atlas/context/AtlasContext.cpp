#include "AtlasContext.h"
#include <iostream>

AtlasContext::AtlasContext(int device_id, const std::string& model_path) 
    : device_id_(device_id) {
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Set device " << device_id_ << " failed!" << std::endl;
        return;
    }

    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Create context failed on device " << device_id_ << std::endl;
        return;
    }

    ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Load model failed: " << model_path << std::endl;
        return;
    }

    model_desc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Get model desc failed!" << std::endl;
        return;
    }

    std::cout << "✅ [Context] Device " << device_id_ << " successfully initialized model." << std::endl;
}

AtlasContext::~AtlasContext() {
    if (model_desc_) {
        aclmdlDestroyDesc(model_desc_);
    }
    if (model_id_ > 0) {
        aclmdlUnload(model_id_);
    }
    if (context_) {
        aclrtDestroyContext(context_);
    }

}