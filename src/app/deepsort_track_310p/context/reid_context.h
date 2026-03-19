#pragma once

#include "framework/context.h"
#include "acl/acl.h"
#include <string>
#include <vector>

class ReidContext : public GryFlux::Context {
public:
    ReidContext(const std::string& model_path, int device_id);
    ~ReidContext() override;

    // 核心接口
    void copyToDevice(const void* data, size_t size);
    void execute();
    std::vector<float> copyToHost(int feature_dim = 512);

    // 框架要求：多线程切换时重新绑定硬件设备
    void bindCurrentThread();

private:
    int device_id_;
    uint32_t model_id_;
    aclmdlDesc* model_desc_;
    
    // 输入输出数据集句柄
    aclmdlDataset* input_dataset_;
    aclmdlDataset* output_dataset_;

    // NPU 显存指针
    void* device_input_ptr_;
    void* device_output_ptr_;
    size_t input_size_;
    size_t output_size_;

    // 内部初始化函数
    void initAclResources(const std::string& model_path);
    void destroyAclResources();
};