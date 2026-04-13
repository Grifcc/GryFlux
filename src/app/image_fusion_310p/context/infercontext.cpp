#include "context/infercontext.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace {

void ThrowIfAclError(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }
    throw std::runtime_error(std::string(action) + " failed, aclError=" + std::to_string(error));
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

FusionModelInfo BuildModelInfo(const InferContext& context, int width, int height) {
    FusionModelInfo model_info;
    model_info.model_width = width;
    model_info.model_height = height;
    model_info.input_element_count = context.GetInputElementCount(0);
    model_info.output_element_count = context.GetOutputElementCount();
    return model_info;
}

}  // namespace

InferContext::InferContext(const std::string& model_path, int device_id)
    : device_id_(device_id) {
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
    ThrowIfAclError(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    ThrowIfAclError(aclrtCreateStream(&stream_), "aclrtCreateStream");

    ThrowIfAclError(aclmdlLoadFromFile(model_path.c_str(), &model_id_), "aclmdlLoadFromFile");
    model_desc_ = ThrowIfNull(aclmdlCreateDesc(), "aclmdlCreateDesc");
    ThrowIfAclError(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

    const size_t input_count = aclmdlGetNumInputs(model_desc_);
    const size_t output_count = aclmdlGetNumOutputs(model_desc_);
    if (input_count != 2 || output_count != 1) {
        throw std::runtime_error("image_fusion_310p expects exactly 2 inputs and 1 output");
    }

    input_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(input)");
    output_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(output)");

    input_buffers_.resize(input_count);
    for (size_t index = 0; index < input_count; ++index) {
        auto& input_buffer = input_buffers_[index];
        input_buffer.byte_size = aclmdlGetInputSizeByIndex(model_desc_, index);
        input_buffer.element_count = input_buffer.byte_size / sizeof(float);
        ThrowIfAclError(
            aclrtMalloc(&input_buffer.device_ptr, input_buffer.byte_size, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(input)");
        aclDataBuffer* data_buffer = ThrowIfNull(
            aclCreateDataBuffer(input_buffer.device_ptr, input_buffer.byte_size),
            "aclCreateDataBuffer(input)");
        ThrowIfAclError(aclmdlAddDatasetBuffer(input_dataset_, data_buffer),
                        "aclmdlAddDatasetBuffer(input)");
    }

    output_buffer_.byte_size = aclmdlGetOutputSizeByIndex(model_desc_, 0);
    output_buffer_.element_count = output_buffer_.byte_size / sizeof(float);
    ThrowIfAclError(
        aclrtMalloc(&output_buffer_.device_ptr, output_buffer_.byte_size, ACL_MEM_MALLOC_HUGE_FIRST),
        "aclrtMalloc(output)");
    ThrowIfAclError(
        aclrtMallocHost(&output_buffer_.host_ptr, output_buffer_.byte_size),
        "aclrtMallocHost(output)");
    aclDataBuffer* output_data_buffer = ThrowIfNull(
        aclCreateDataBuffer(output_buffer_.device_ptr, output_buffer_.byte_size),
        "aclCreateDataBuffer(output)");
    ThrowIfAclError(aclmdlAddDatasetBuffer(output_dataset_, output_data_buffer),
                    "aclmdlAddDatasetBuffer(output)");
}

InferContext::~InferContext() {
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

void InferContext::bindCurrentThread() {
    ThrowIfAclError(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
}

size_t InferContext::GetInputElementCount(size_t index) const {
    return input_buffers_.at(index).element_count;
}

size_t InferContext::GetOutputElementCount() const {
    return output_buffer_.element_count;
}

void InferContext::copyInputToDevice(size_t index, const float* host_data, size_t element_count) {
    auto& input_buffer = input_buffers_.at(index);
    const size_t byte_count = element_count * sizeof(float);
    if (byte_count != input_buffer.byte_size) {
        throw std::runtime_error("Fusion input element count does not match model binding size");
    }
    ThrowIfAclError(
        aclrtMemcpy(input_buffer.device_ptr,
                    input_buffer.byte_size,
                    host_data,
                    byte_count,
                    ACL_MEMCPY_HOST_TO_DEVICE),
        "aclrtMemcpy(input)");
}

void InferContext::execute() {
    ThrowIfAclError(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "aclmdlExecute");
}

void InferContext::copyOutputToHost(float* host_data, size_t element_count) {
    const size_t byte_count = element_count * sizeof(float);
    if (byte_count > output_buffer_.byte_size) {
        throw std::runtime_error("Fusion output element count exceeds model output size");
    }
    ThrowIfAclError(
        aclrtMemcpy(output_buffer_.host_ptr,
                    output_buffer_.byte_size,
                    output_buffer_.device_ptr,
                    output_buffer_.byte_size,
                    ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(output)");
    std::memcpy(host_data, output_buffer_.host_ptr, byte_count);
}

void InferContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    input_dataset_ = nullptr;
    DestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
}

void InferContext::destroyBuffers() noexcept {
    for (auto& input_buffer : input_buffers_) {
        if (input_buffer.device_ptr != nullptr) {
            aclrtFree(input_buffer.device_ptr);
            input_buffer.device_ptr = nullptr;
        }
        if (input_buffer.host_ptr != nullptr) {
            aclrtFreeHost(input_buffer.host_ptr);
            input_buffer.host_ptr = nullptr;
        }
    }
    input_buffers_.clear();

    if (output_buffer_.device_ptr != nullptr) {
        aclrtFree(output_buffer_.device_ptr);
        output_buffer_.device_ptr = nullptr;
    }
    if (output_buffer_.host_ptr != nullptr) {
        aclrtFreeHost(output_buffer_.host_ptr);
        output_buffer_.host_ptr = nullptr;
    }
    output_buffer_ = TensorBuffer{};
}

FusionInferResourceBundle CreateFusionInferResourceBundle(
    const std::string& model_path,
    int device_id,
    size_t instance_count,
    int fallback_width,
    int fallback_height) {
    if (instance_count == 0) {
        throw std::runtime_error("Fusion ACL context instance count must be greater than zero");
    }

    auto first_context = std::make_shared<InferContext>(model_path, device_id);
    FusionInferResourceBundle bundle;
    bundle.model_info = BuildModelInfo(*first_context, fallback_width, fallback_height);

    const size_t expected_input_elements =
        static_cast<size_t>(fallback_width) * static_cast<size_t>(fallback_height);
    const size_t expected_output_elements = expected_input_elements;
    if (bundle.model_info.input_element_count != expected_input_elements ||
        bundle.model_info.output_element_count != expected_output_elements) {
        throw std::runtime_error(
            "Fusion model IO size does not match fallback width/height; pass correct --width/--height");
    }

    bundle.contexts.reserve(instance_count);
    bundle.contexts.push_back(first_context);
    for (size_t index = 1; index < instance_count; ++index) {
        bundle.contexts.push_back(std::make_shared<InferContext>(model_path, device_id));
    }
    return bundle;
}
