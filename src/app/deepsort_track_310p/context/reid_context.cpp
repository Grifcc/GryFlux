#include "reid_context.h"
#include <iostream>

ReidContext::ReidContext(const std::string& model_path, int device_id) 
    : device_id_(device_id), model_id_(0), model_desc_(nullptr),
      input_dataset_(nullptr), output_dataset_(nullptr),
      device_input_ptr_(nullptr), device_output_ptr_(nullptr) {
    
    initAclResources(model_path);
}

ReidContext::~ReidContext() {
    destroyAclResources();
}

void ReidContext::bindCurrentThread() {
    // 关键：在多线程中，必须显式调用此函数，否则后续 ACL 调用会找不到 Device
    aclrtSetDevice(device_id_);
}

void ReidContext::initAclResources(const std::string& model_path) {
    aclrtSetDevice(device_id_);

    // 1. 加载模型
    aclmdlLoadFromFile(model_path.c_str(), &model_id_);
    model_desc_ = aclmdlCreateDesc();
    aclmdlGetDesc(model_desc_, model_id_);

    // 2. 准备输入内存 (假设模型输入是固定的 128x256x3)
    input_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
    aclrtMalloc(&device_input_ptr_, input_size_, ACL_MEM_MALLOC_NORMAL_ONLY);
    input_dataset_ = aclmdlCreateDataset();
    aclDataBuffer* input_buffer = aclCreateDataBuffer(device_input_ptr_, input_size_);
    aclmdlAddDatasetBuffer(input_dataset_, input_buffer);

    // 3. 准备输出内存 (通常是 512 维特征向量)
    output_size_ = aclmdlGetOutputSizeByIndex(model_desc_, 0);
    aclrtMalloc(&device_output_ptr_, output_size_, ACL_MEM_MALLOC_NORMAL_ONLY);
    output_dataset_ = aclmdlCreateDataset();
    aclDataBuffer* output_buffer = aclCreateDataBuffer(device_output_ptr_, output_size_);
    aclmdlAddDatasetBuffer(output_dataset_, output_buffer);
}

void ReidContext::copyToDevice(const void* data, size_t size) {
    // 将 CPU 上的 NCHW 数据搬运到 NPU 推理区
    aclrtMemcpy(device_input_ptr_, input_size_, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
}

void ReidContext::execute() {
    // 触发硬件推理 (同步调用)
    aclmdlExecute(model_id_, input_dataset_, output_dataset_);
}

std::vector<float> ReidContext::copyToHost(int feature_dim) {
    std::vector<float> result(feature_dim);
    aclrtMemcpy(result.data(), output_size_, device_output_ptr_, output_size_, ACL_MEMCPY_DEVICE_TO_HOST);
    return result;
}

void ReidContext::destroyAclResources() {
    // 严格按照初始化逆序释放，防止内存泄漏
    if (device_input_ptr_) aclrtFree(device_input_ptr_);
    if (device_output_ptr_) aclrtFree(device_output_ptr_);
    if (input_dataset_) aclmdlDestroyDataset(input_dataset_);
    if (output_dataset_) aclmdlDestroyDataset(output_dataset_);
    if (model_desc_) aclmdlDestroyDesc(model_desc_);
    if (model_id_) aclmdlUnload(model_id_);
}