#include "AtlasContext.h"

#include "acl_environment.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

void SetError(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
}

void CheckAcl(aclError error, const char* action) {
    if (error == ACL_SUCCESS) {
        return;
    }

    throw std::runtime_error(
        std::string(action) + " failed, aclError=" + std::to_string(error));
}

template <typename T>
T* CheckNotNull(T* pointer, const char* action) {
    if (pointer != nullptr) {
        return pointer;
    }

    throw std::runtime_error(std::string(action) + " returned null");
}

void DestroyDataset(aclmdlDataset*& dataset) noexcept {
    if (dataset == nullptr) {
        return;
    }

    const size_t buffer_count = aclmdlGetDatasetNumBuffers(dataset);
    for (size_t index = 0; index < buffer_count; ++index) {
        if (aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(dataset, index)) {
            aclDestroyDataBuffer(data_buffer);
        }
    }

    aclmdlDestroyDataset(dataset);
    dataset = nullptr;
}

}  // namespace

AtlasContext::AtlasContext(std::string model_path, int device_id)
    : model_path_(std::move(model_path)),
      device_id_(device_id) {}

AtlasContext::~AtlasContext() {
    cleanup();
}

bool AtlasContext::init(std::string* error) {
    if (initialized_) {
        return true;
    }

    if (!ZeroDce::AclEnvironment::acquire(device_id_, error)) {
        return false;
    }
    acl_ready_ = true;

    try {
        setDevice();
        loadModel();
        createInputBuffers();
        createOutputBuffers();
        initialized_ = true;
        return true;
    } catch (const std::exception& exception) {
        SetError(error, exception.what());
        cleanup();
        return false;
    }
}

void AtlasContext::run(const void* input_data, size_t input_size) {
    if (!initialized_) {
        throw std::runtime_error("AtlasContext is not initialized");
    }
    if (input_buffers_.empty()) {
        throw std::runtime_error("Model input count is zero");
    }

    ModelInput& input = input_buffers_.front();
    if (input_size != input.size) {
        throw std::runtime_error("Input buffer size mismatch");
    }

    setDevice();
    CheckAcl(aclrtMemcpy(input.device_buffer,
                         input.size,
                         input_data,
                         input_size,
                         ACL_MEMCPY_HOST_TO_DEVICE),
             "aclrtMemcpy(host_to_device)");
    CheckAcl(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "aclmdlExecute");

    for (auto& output : output_buffers_) {
        CheckAcl(aclrtMemcpy(output.host_buffer,
                             output.size,
                             output.device_buffer,
                             output.size,
                             ACL_MEMCPY_DEVICE_TO_HOST),
                 "aclrtMemcpy(device_to_host)");
    }
}

const void* AtlasContext::getOutputHostBuffer(size_t index) const {
    return outputBuffer(index).host_buffer;
}

size_t AtlasContext::getOutputSize(size_t index) const {
    return outputBuffer(index).size;
}

void AtlasContext::setDevice() const {
    CheckAcl(aclrtSetDevice(device_id_), "aclrtSetDevice");
}

void AtlasContext::loadModel() {
    CheckAcl(aclmdlLoadFromFile(model_path_.c_str(), &model_id_), "aclmdlLoadFromFile");
    model_loaded_ = true;

    model_desc_ = CheckNotNull(aclmdlCreateDesc(), "aclmdlCreateDesc");
    CheckAcl(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");
}

void AtlasContext::createInputBuffers() {
    input_dataset_ = CheckNotNull(aclmdlCreateDataset(), "aclmdlCreateDataset(input)");

    const size_t input_count = aclmdlGetNumInputs(model_desc_);
    input_buffers_.reserve(input_count);
    for (size_t index = 0; index < input_count; ++index) {
        ModelInput input;
        input.size = aclmdlGetInputSizeByIndex(model_desc_, index);
        CheckAcl(aclrtMalloc(&input.device_buffer, input.size, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(input)");

        aclDataBuffer* buffer = CheckNotNull(
            aclCreateDataBuffer(input.device_buffer, input.size),
            "aclCreateDataBuffer(input)");
        CheckAcl(aclmdlAddDatasetBuffer(input_dataset_, buffer),
                 "aclmdlAddDatasetBuffer(input)");
        input_buffers_.push_back(input);
    }
}

void AtlasContext::createOutputBuffers() {
    output_dataset_ = CheckNotNull(aclmdlCreateDataset(), "aclmdlCreateDataset(output)");

    const size_t output_count = aclmdlGetNumOutputs(model_desc_);
    output_buffers_.reserve(output_count);
    for (size_t index = 0; index < output_count; ++index) {
        ModelOutput output;
        output.size = aclmdlGetOutputSizeByIndex(model_desc_, index);

        CheckAcl(aclrtMalloc(&output.device_buffer, output.size, ACL_MEM_MALLOC_HUGE_FIRST),
                 "aclrtMalloc(output)");
        CheckAcl(aclrtMallocHost(&output.host_buffer, output.size),
                 "aclrtMallocHost(output)");

        aclDataBuffer* buffer = CheckNotNull(
            aclCreateDataBuffer(output.device_buffer, output.size),
            "aclCreateDataBuffer(output)");
        CheckAcl(aclmdlAddDatasetBuffer(output_dataset_, buffer),
                 "aclmdlAddDatasetBuffer(output)");
        output_buffers_.push_back(output);
    }
}

const AtlasContext::ModelOutput& AtlasContext::outputBuffer(size_t index) const {
    if (index >= output_buffers_.size()) {
        throw std::runtime_error("Output index is out of range");
    }
    return output_buffers_[index];
}

void AtlasContext::cleanup() noexcept {
    if (acl_ready_) {
        aclrtSetDevice(device_id_);
    }

    destroyDatasets();
    destroyBuffers();
    unloadModel();
    initialized_ = false;

    if (acl_ready_) {
        ZeroDce::AclEnvironment::release(device_id_);
        acl_ready_ = false;
    }
}

void AtlasContext::destroyDatasets() noexcept {
    DestroyDataset(input_dataset_);
    DestroyDataset(output_dataset_);
}

void AtlasContext::destroyBuffers() noexcept {
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

void AtlasContext::unloadModel() noexcept {
    if (model_desc_ != nullptr) {
        aclmdlDestroyDesc(model_desc_);
        model_desc_ = nullptr;
    }

    if (model_loaded_) {
        aclmdlUnload(model_id_);
        model_id_ = 0;
        model_loaded_ = false;
    }
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateAtlasContexts(
    const std::string& om_model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("Atlas context instance count must be greater than zero");
    }

    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        auto context = std::make_shared<AtlasContext>(om_model_path, device_id);
        std::string init_error;
        if (!context->init(&init_error)) {
            throw std::runtime_error("AtlasContext init failed: " + init_error);
        }
        contexts.push_back(std::move(context));
    }
    return contexts;
}
