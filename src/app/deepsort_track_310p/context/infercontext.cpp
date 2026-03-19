#include "infercontext.h"
#include <stdexcept>
#include <iostream>

// 异常检查宏 (抛出异常以符合 RAII 构造失败，防止程序带着坏状态运行)
#define ACL_CHECK_THROW(op, error_msg) \
    do { \
        aclError ret = (op); \
        if (ret != ACL_SUCCESS) { \
            throw std::runtime_error(std::string(error_msg) + " failed! ret=" + std::to_string(ret)); \
        } \
    } while (0)

InferContext::InferContext(const std::string& om_model_path, int device_id)
    : device_id_(device_id) 
{
    std::cout << "[InferContext] 开始初始化 Ascend 资源, Device: " << device_id_ << std::endl;

    ACL_CHECK_THROW(aclrtSetDevice(device_id_), "设置Device失败");
    ACL_CHECK_THROW(aclrtCreateContext(&context_, device_id_), "创建Context失败");
    ACL_CHECK_THROW(aclrtSetCurrentContext(context_), "设置Current Context失败");
    ACL_CHECK_THROW(aclrtCreateStream(&stream_), "创建Stream失败");

    ACL_CHECK_THROW(aclmdlLoadFromFile(om_model_path.c_str(), &model_id_), "加载OM模型失败");
    model_desc_ = aclmdlCreateDesc();
    ACL_CHECK_THROW(aclmdlGetDesc(model_desc_, model_id_), "获取模型描述失败");

    input_buffer_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
    ACL_CHECK_THROW(aclrtMalloc(&input_buffer_, input_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST), "分配设备输入内存失败");
    
    input_dataset_ = aclmdlCreateDataset();
    auto input_data_buf = aclCreateDataBuffer(input_buffer_, input_buffer_size_);
    ACL_CHECK_THROW(aclmdlAddDatasetBuffer(input_dataset_, input_data_buf), "添加 Input Buffer 失败");

    size_t output_num = aclmdlGetNumOutputs(model_desc_);
    output_buffers_.resize(output_num);
    output_dataset_ = aclmdlCreateDataset();

    for (size_t i = 0; i < output_num; ++i) {
        auto& out = output_buffers_[i];
        out.size = aclmdlGetOutputSizeByIndex(model_desc_, i);
        
        ACL_CHECK_THROW(aclrtMalloc(&out.device_buffer, out.size, ACL_MEM_MALLOC_HUGE_FIRST), "分配设备输出内存失败");
        ACL_CHECK_THROW(aclrtMallocHost(&out.host_buffer, out.size), "分配主机输出内存失败");
        
        auto output_data_buf = aclCreateDataBuffer(out.device_buffer, out.size);
        ACL_CHECK_THROW(aclmdlAddDatasetBuffer(output_dataset_, output_data_buf), "添加 Output Buffer 失败");
    }
    
    std::cout << "[InferContext] Ascend 资源初始化成功！" << std::endl;
}

InferContext::~InferContext() {
    std::cout << "[InferContext] 开始释放 Ascend 资源..." << std::endl;

    if (input_buffer_) aclrtFree(input_buffer_);
    if (input_dataset_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_dataset_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(input_dataset_, i));
        }
        aclmdlDestroyDataset(input_dataset_);
    }

    for (auto& out : output_buffers_) {
        if (out.device_buffer) aclrtFree(out.device_buffer);
        if (out.host_buffer) aclrtFreeHost(out.host_buffer);
    }
    if (output_dataset_) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_dataset_); ++i) {
            aclDestroyDataBuffer(aclmdlGetDatasetBuffer(output_dataset_, i));
        }
        aclmdlDestroyDataset(output_dataset_);
    }

    if (model_desc_) aclmdlDestroyDesc(model_desc_);
    if (model_id_ != 0) aclmdlUnload(model_id_);
    if (stream_) aclrtDestroyStream(stream_);
    if (context_) aclrtDestroyContext(context_);
    
    std::cout << "[InferContext] Ascend 资源释放完毕" << std::endl;
}

void InferContext::bindCurrentThread() {
    aclError ret = aclrtSetCurrentContext(context_);
    if (ret != ACL_SUCCESS) {
        std::cerr << "[InferContext] 警告: bindCurrentThread 失败! ret=" << ret << std::endl;
    }
}

void InferContext::copyToDevice(const void* host_data, size_t size) {
    ACL_CHECK_THROW(aclrtMemcpy(input_buffer_, input_buffer_size_, host_data, size, ACL_MEMCPY_HOST_TO_DEVICE), "Host to Device 失败");
}

void InferContext::executeModel() {
    ACL_CHECK_THROW(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "模型推理失败");
}

void InferContext::copyToHost() {
    for (auto& out_buf : output_buffers_) {
        ACL_CHECK_THROW(aclrtMemcpy(out_buf.host_buffer, out_buf.size, out_buf.device_buffer, out_buf.size, ACL_MEMCPY_DEVICE_TO_HOST), "Device to Host 失败");
    }
}