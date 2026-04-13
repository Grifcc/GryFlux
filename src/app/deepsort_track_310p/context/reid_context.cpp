#include "context/reid_context.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

void ThrowIfAclError(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string(action) + " failed, aclError=" + std::to_string(error));
}

template <typename T>
T* ThrowIfNull(T* pointer, const char* action) {
    if (pointer != nullptr) {
        return pointer;
    }

    throw std::runtime_error(std::string(action) + " returned null");
}

void DestroyDataset(aclmdlDataset* dataset) noexcept {
    if (dataset == nullptr) {
        return;
    }

    const size_t buffer_count = aclmdlGetDatasetNumBuffers(dataset);
    for (size_t index = 0; index < buffer_count; ++index) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(dataset, index);
        if (data_buffer != nullptr) {
            aclDestroyDataBuffer(data_buffer);
        }
    }
    aclmdlDestroyDataset(dataset);
}

}  // namespace

ReidContext::ReidContext(const std::string& model_path, int device_id)
    : device_id_(device_id) {
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
    ThrowIfAclError(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    ThrowIfAclError(aclrtCreateStream(&stream_), "aclrtCreateStream");

    ThrowIfAclError(aclmdlLoadFromFile(model_path.c_str(), &model_id_),
                    "aclmdlLoadFromFile(reid)");
    model_desc_ = ThrowIfNull(aclmdlCreateDesc(), "aclmdlCreateDesc(reid)");
    ThrowIfAclError(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc(reid)");

    input_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
    output_size_ = aclmdlGetOutputSizeByIndex(model_desc_, 0);

    ThrowIfAclError(
        aclrtMalloc(&device_input_ptr_, input_size_, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(reid_input)");
    ThrowIfAclError(
        aclrtMalloc(&device_output_ptr_, output_size_, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(reid_output)");
    ThrowIfAclError(
        aclrtMallocHost(&host_output_ptr_, output_size_),
        "aclrtMallocHost(reid_output)");

    input_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(reid_input)");
    output_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(reid_output)");

    aclDataBuffer* input_buffer = ThrowIfNull(
        aclCreateDataBuffer(device_input_ptr_, input_size_),
        "aclCreateDataBuffer(reid_input)");
    aclDataBuffer* output_buffer = ThrowIfNull(
        aclCreateDataBuffer(device_output_ptr_, output_size_),
        "aclCreateDataBuffer(reid_output)");

    ThrowIfAclError(
        aclmdlAddDatasetBuffer(input_dataset_, input_buffer),
        "aclmdlAddDatasetBuffer(reid_input)");
    ThrowIfAclError(
        aclmdlAddDatasetBuffer(output_dataset_, output_buffer),
        "aclmdlAddDatasetBuffer(reid_output)");
}

ReidContext::~ReidContext() {
    if (context_ != nullptr) {
        aclrtSetCurrentContext(context_);
    }

    destroyDatasets();
    destroyBuffers();

    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
    if (stream_ != nullptr) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_ != nullptr) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
}

void ReidContext::bindCurrentThread() {
    ThrowIfAclError(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext(reid)");
}

void ReidContext::copyToDevice(const void* data, size_t size) {
    if (size > input_size_) {
        throw std::runtime_error("ReID input buffer size exceeds model input allocation");
    }

    ThrowIfAclError(
        aclrtMemcpy(device_input_ptr_, input_size_, data, size, ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(reid_host_to_device)");
}

void ReidContext::execute() {
    ThrowIfAclError(
        aclmdlExecute(model_id_, input_dataset_, output_dataset_),
        "aclmdlExecute(reid)");
}

void ReidContext::copyToHost(float* host_output, size_t element_count) {
    const size_t requested_bytes = element_count * sizeof(float);
    if (requested_bytes > output_size_) {
        throw std::runtime_error("Requested ReID output exceeds model output allocation");
    }

    ThrowIfAclError(
        aclrtMemcpy(host_output_ptr_, output_size_, device_output_ptr_, output_size_, ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(reid_device_to_host)");
    std::copy_n(static_cast<float*>(host_output_ptr_), element_count, host_output);
}

void ReidContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    input_dataset_ = nullptr;

    DestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
}

void ReidContext::destroyBuffers() noexcept {
    if (device_input_ptr_ != nullptr) {
        aclrtFree(device_input_ptr_);
        device_input_ptr_ = nullptr;
    }
    if (device_output_ptr_ != nullptr) {
        aclrtFree(device_output_ptr_);
        device_output_ptr_ = nullptr;
    }
    if (host_output_ptr_ != nullptr) {
        aclrtFreeHost(host_output_ptr_);
        host_output_ptr_ = nullptr;
    }
    input_size_ = 0;
    output_size_ = 0;
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateReidInferContexts(
    const std::string& model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("ACL ReID context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        contexts.push_back(std::make_shared<ReidContext>(model_path, device_id));
    }
    return contexts;
}
