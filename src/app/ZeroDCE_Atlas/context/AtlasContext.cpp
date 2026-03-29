#include "AtlasContext.h"
#include <iostream>

AtlasContext::AtlasContext(int device_id, const std::string& model_path) 
    : device_id_(device_id) {
    // 1. 指定当前操作的 NPU 设备
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Set device " << device_id_ << " failed!" << std::endl;
        return;
    }

    // 2. 创建 Context (非常关键，每个推理线程必须绑定它)
    ret = aclrtCreateContext(&context_, device_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Create context failed on device " << device_id_ << std::endl;
        return;
    }

    // 3. 加载 OM 模型并获取 Model ID
    ret = aclmdlLoadFromFile(model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Load model failed: " << model_path << std::endl;
        return;
    }

    // 4. 创建并获取模型的描述信息 (输入输出的 Tensor 尺寸等)
    model_desc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(model_desc_, model_id_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "❌ [Context] Get model desc failed!" << std::endl;
        return;
    }

    std::cout << "✅ [Context] Device " << device_id_ << " successfully initialized model." << std::endl;
}

AtlasContext::~AtlasContext() {
    // 严格按照与初始化相反的顺序释放资源
    if (model_desc_) {
        aclmdlDestroyDesc(model_desc_);
    }
    if (model_id_ > 0) {
        aclmdlUnload(model_id_);
    }
    if (context_) {
        aclrtDestroyContext(context_);
    }
    // 注意：通常不由 Context 去 ResetDevice，而是在主程序 aclFinalize() 之前统一做
}