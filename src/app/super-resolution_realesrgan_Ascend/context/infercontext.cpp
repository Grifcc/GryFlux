#include "context/infercontext.h"

#include "context/AclEnvironment.h"

#include <stdexcept>
#include <string>
#include <utility>

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

InferContext::InferContext(Config config)
    : config_(std::move(config)),
      device_id_(config_.device_id) {}

InferContext::~InferContext() {
    if (initialized_) {
        aclrtSetDevice(device_id_);
    }
    destroyDatasets();
    destroyBuffers();
    unloadModel();
    if (initialized_) {
        YoloxRuntime::AclEnvironment::release(device_id_);
    }
}

bool InferContext::init(std::string* error) {
    if (initialized_) {
        return true;
    }

    if (!YoloxRuntime::AclEnvironment::acquire(device_id_, error)) {
        return false;
    }

    try {
        ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");

        ThrowIfAclError(
            aclmdlLoadFromFile(config_.model_path.c_str(), &model_id_),
            "aclmdlLoadFromFile");
        model_desc_ = ThrowIfNull(aclmdlCreateDesc(), "aclmdlCreateDesc");
        ThrowIfAclError(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

        input_buffer_size_ = aclmdlGetInputSizeByIndex(model_desc_, 0);
        ThrowIfAclError(
            aclrtMalloc(&input_buffer_, input_buffer_size_, ACL_MEM_MALLOC_HUGE_FIRST),
            "aclrtMalloc(input)");

        input_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(input)");
        aclDataBuffer* input_data_buffer = ThrowIfNull(
            aclCreateDataBuffer(input_buffer_, input_buffer_size_),
            "aclCreateDataBuffer(input)");
        ThrowIfAclError(
            aclmdlAddDatasetBuffer(input_dataset_, input_data_buffer),
            "aclmdlAddDatasetBuffer(input)");

        output_dataset_ = ThrowIfNull(aclmdlCreateDataset(), "aclmdlCreateDataset(output)");
        const size_t output_count = aclmdlGetNumOutputs(model_desc_);
        output_buffers_.reserve(output_count);
        for (size_t index = 0; index < output_count; ++index) {
            ModelOutput output;
            output.size = aclmdlGetOutputSizeByIndex(model_desc_, index);
            output.format = aclmdlGetOutputFormat(model_desc_, index);
            output.data_type = aclmdlGetOutputDataType(model_desc_, index);
            ThrowIfAclError(
                aclmdlGetOutputDims(model_desc_, index, &output.dims),
                "aclmdlGetOutputDims");

            ThrowIfAclError(
                aclrtMalloc(&output.device_buffer,
                            output.size,
                            ACL_MEM_MALLOC_HUGE_FIRST),
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
    } catch (const std::exception& e) {
        SetError(error, e.what());
        destroyDatasets();
        destroyBuffers();
        unloadModel();
        YoloxRuntime::AclEnvironment::release(device_id_);
        return false;
    }

    initialized_ = true;
    return true;
}

void InferContext::copyToDevice(const void* host_data, size_t size) {
    if (size > input_buffer_size_) {
        throw std::runtime_error("Input buffer size exceeds model input allocation");
    }

    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(aclrtMemcpy(input_buffer_,
                                input_buffer_size_,
                                host_data,
                                size,
                                ACL_MEMCPY_HOST_TO_DEVICE),
                    "aclrtMemcpy(host_to_device)");
}

void InferContext::executeModel() {
    ThrowIfAclError(aclrtSetDevice(device_id_), "aclrtSetDevice");
    ThrowIfAclError(
        aclmdlExecute(model_id_, input_dataset_, output_dataset_),
        "aclmdlExecute");
}

void InferContext::copyToHost() {
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

void InferContext::unloadModel() noexcept {
    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }
    if (model_id_ != 0) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
    }
}

void InferContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    input_dataset_ = nullptr;

    DestroyDataset(output_dataset_);
    output_dataset_ = nullptr;
}

void InferContext::destroyBuffers() noexcept {
    if (input_buffer_ != nullptr) {
        aclrtFree(input_buffer_);
        input_buffer_ = nullptr;
    }

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
    input_buffer_size_ = 0;
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateInferContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("ACL context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        auto context = std::make_shared<InferContext>(InferContext::Config{om_model_path, device_id});
        std::string init_error;
        if (!context->init(&init_error)) {
            throw std::runtime_error("InferContext init failed: " + init_error);
        }
        contexts.push_back(std::move(context));
    }
    return contexts;
}
