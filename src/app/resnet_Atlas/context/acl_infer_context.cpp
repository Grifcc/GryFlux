#include "context/acl_infer_context.h"

#include "context/acl_environment.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace resnet {
namespace {

void ThrowIfAclError(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string(action) + " failed, aclError=" + std::to_string(error));
}

void SetError(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
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

AclInferContext::AclInferContext(Config config)
    : config_(std::move(config)),
      device_id_(config_.device_id) {}

AclInferContext::~AclInferContext() {
    if (initialized_) {
        aclrtSetDevice(device_id_);
    }
    destroyDatasets();
    destroyBuffers();
    unloadModel();
    if (initialized_) {
        AclEnvironment::release(device_id_);
    }
}

bool AclInferContext::init(std::string* error) {
    if (initialized_) {
        return true;
    }

    if (!AclEnvironment::acquire(device_id_, error)) {
        return false;
    }

    try {
        ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");

        ThrowIfAclError(aclmdlLoadFromFile(config_.model_path.c_str(), &model_id_),
                        "aclmdlLoadFromFile");
        model_desc_ = ThrowIfNull(aclmdlCreateDesc(), "aclmdlCreateDesc");
        ThrowIfAclError(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

        input_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(input)");
        const size_t input_count = aclmdlGetNumInputs(model_desc_);
        input_buffers_.reserve(input_count);
        for (size_t index = 0; index < input_count; ++index) {
            ModelInput input;
            input.size = aclmdlGetInputSizeByIndex(model_desc_, index);
            ThrowIfAclError(
                aclrtMalloc(&input.device_buffer, input.size, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(input)");

            aclDataBuffer* input_data_buffer = ThrowIfNull(
                aclCreateDataBuffer(input.device_buffer, input.size),
                "aclCreateDataBuffer(input)");
            ThrowIfAclError(
                aclmdlAddDatasetBuffer(input_dataset_, input_data_buffer),
                "aclmdlAddDatasetBuffer(input)");
            input_buffers_.push_back(input);
        }

        output_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(output)");
        const size_t output_count = aclmdlGetNumOutputs(model_desc_);
        output_buffers_.reserve(output_count);
        for (size_t index = 0; index < output_count; ++index) {
            ModelOutput output;
            output.size = aclmdlGetOutputSizeByIndex(model_desc_, index);

            ThrowIfAclError(
                aclrtMalloc(&output.device_buffer, output.size, ACL_MEM_MALLOC_HUGE_FIRST),
                "aclrtMalloc(output)");
            ThrowIfAclError(
                aclrtMallocHost(&output.host_buffer, output.size),
                "aclrtMallocHost(output)");

            aclDataBuffer* output_data_buffer = ThrowIfNull(
                aclCreateDataBuffer(output.device_buffer, output.size),
                "aclCreateDataBuffer(output)");
            ThrowIfAclError(
                aclmdlAddDatasetBuffer(output_dataset_, output_data_buffer),
                "aclmdlAddDatasetBuffer(output)");
            output_buffers_.push_back(output);
        }
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
        destroyDatasets();
        destroyBuffers();
        unloadModel();
        AclEnvironment::release(device_id_);
        return false;
    }

    initialized_ = true;
    return true;
}

void AclInferContext::copyToDevice(const void* host_data, size_t size) {
    copyToDevice(0, host_data, size);
}

void AclInferContext::copyToDevice(size_t input_index,
                                   const void* host_data,
                                   size_t size) {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }

    const ModelInput& input = input_buffers_[input_index];
    if (size > input.size) {
        throw std::runtime_error("Input buffer size exceeds model input allocation");
    }

    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(aclrtMemcpy(input.device_buffer,
                                input.size,
                                host_data,
                                size,
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(host_to_device)");
}

void AclInferContext::executeModel() {
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(
        aclmdlExecute(model_id_, input_dataset_, output_dataset_),
        "aclmdlExecute");
}

void AclInferContext::copyToHost() {
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    for (auto& output : output_buffers_) {
        ThrowIfAclError(
            aclrtMemcpy(output.host_buffer,
                        output.size,
                        output.device_buffer,
                        output.size,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "aclrtMemcpy(device_to_host)");
    }
}

void AclInferContext::copyToHost(size_t output_index,
                                 void* host_buffer,
                                 size_t size) {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }

    ModelOutput& output = output_buffers_[output_index];
    if (size > output.size) {
        throw std::runtime_error("Requested output size exceeds model output allocation");
    }

    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(
        aclrtMemcpy(host_buffer,
                    size,
                    output.device_buffer,
                    size,
                    ACL_MEMCPY_DEVICE_TO_HOST),
        "aclrtMemcpy(device_to_host_direct)");
}

aclmdlIODims AclInferContext::getInputDims(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }

    aclmdlIODims dims{};
    ThrowIfAclError(aclmdlGetInputDims(model_desc_, input_index, &dims), "aclmdlGetInputDims");
    return dims;
}

aclmdlIODims AclInferContext::getOutputDims(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }

    aclmdlIODims dims{};
    ThrowIfAclError(aclmdlGetOutputDims(model_desc_, output_index, &dims), "aclmdlGetOutputDims");
    return dims;
}

aclmdlIODims AclInferContext::getCurrentOutputDims(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }

    aclmdlIODims dims{};
    ThrowIfAclError(
        aclmdlGetCurOutputDims(model_desc_, output_index, &dims),
        "aclmdlGetCurOutputDims");
    return dims;
}

aclFormat AclInferContext::getInputFormat(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }
    return aclmdlGetInputFormat(model_desc_, input_index);
}

aclFormat AclInferContext::getOutputFormat(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return aclmdlGetOutputFormat(model_desc_, output_index);
}

aclDataType AclInferContext::getInputDataType(size_t input_index) const {
    if (input_index >= input_buffers_.size()) {
        throw std::runtime_error("Input index is out of range");
    }
    return aclmdlGetInputDataType(model_desc_, input_index);
}

aclDataType AclInferContext::getOutputDataType(size_t output_index) const {
    if (output_index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return aclmdlGetOutputDataType(model_desc_, output_index);
}

void AclInferContext::unloadModel() noexcept {
    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
}

void AclInferContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    input_dataset_ = nullptr;

    DestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
}

void AclInferContext::destroyBuffers() noexcept {
    for (auto& input : input_buffers_) {
        if (input.device_buffer != nullptr) {
            aclrtFree(input.device_buffer);
            input.device_buffer = nullptr;
        }
    }
    input_buffers_.clear();

    for (auto& output : output_buffers_) {
        if (output.device_buffer != nullptr) {
            aclrtFree(output.device_buffer);
            output.device_buffer = nullptr;
        }
        if (output.host_buffer != nullptr) {
            aclrtFreeHost(output.host_buffer);
            output.host_buffer = nullptr;
        }
    }
    output_buffers_.clear();
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateAclInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("ACL context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        auto context = std::make_shared<AclInferContext>(
            AclInferContext::Config{om_model_path, device_id});
        std::string init_error;
        if (!context->init(&init_error)) {
            throw std::runtime_error("AclInferContext init failed: " + init_error);
        }
        contexts.push_back(std::move(context));
    }
    return contexts;
}

}  // namespace resnet
