#include "reid_context.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#define CUDA_CHECK_THROW(op, error_msg) \
    do { \
        cudaError_t ret = (op); \
        if (ret != cudaSuccess) { \
            throw std::runtime_error(std::string(error_msg) + " failed! ret=" + std::to_string(ret)); \
        } \
    } while (0)

#define TRT_CHECK_THROW(op, error_msg) \
    do { \
        if (!(op)) { \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

namespace {

template <typename T>
struct TrtDeleter {
    void operator()(T* ptr) const {
        if (!ptr) {
            return;
        }
#if NV_TENSORRT_MAJOR >= 10
        delete ptr;
#else
        ptr->destroy();
#endif
    }
};

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > Severity::kWARNING) {
            return;
        }
        std::cerr << "[TensorRT] " << msg << std::endl;
    }
};

TrtLogger& trtLogger() {
    static TrtLogger logger;
    return logger;
}

size_t elementSize(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kBOOL:
            return 1;
#if NV_TENSORRT_MAJOR >= 8
        case nvinfer1::DataType::kINT64:
            return 8;
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT data type");
    }
}

size_t volume(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("Dynamic TensorRT shapes are not supported by this ReID sample");
        }
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

size_t bindingSize(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    const char* tensor_name = engine.getIOTensorName(binding_index);
    return volume(engine.getTensorShape(tensor_name)) *
           elementSize(engine.getTensorDataType(tensor_name));
#else
    return volume(engine.getBindingDimensions(binding_index)) *
           elementSize(engine.getBindingDataType(binding_index));
#endif
}

bool bindingIsInput(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorIOMode(engine.getIOTensorName(binding_index)) == nvinfer1::TensorIOMode::kINPUT;
#else
    return engine.bindingIsInput(binding_index);
#endif
}

int bindingCount(const nvinfer1::ICudaEngine& engine) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getNbIOTensors();
#else
    return engine.getNbBindings();
#endif
}

bool enqueueContext(nvinfer1::IExecutionContext& context, std::vector<void*>& bindings, cudaStream_t stream) {
#if NV_TENSORRT_MAJOR >= 10
    for (int i = 0; i < context.getEngine().getNbIOTensors(); ++i) {
        const char* tensor_name = context.getEngine().getIOTensorName(i);
        if (!context.setTensorAddress(tensor_name, bindings[i])) {
            return false;
        }
    }
    return context.enqueueV3(stream);
#else
    return context.enqueueV2(bindings.data(), stream, nullptr);
#endif
}

}  // namespace

class SharedReidModel {
public:
    explicit SharedReidModel(const std::string& engine_path) {
        std::ifstream input(engine_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Failed to open ReID TensorRT engine file: " + engine_path);
        }

        input.seekg(0, std::ios::end);
        const std::streamsize engine_size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (engine_size <= 0) {
            throw std::runtime_error(
                "ReID TensorRT engine file is empty: " + engine_path);
        }

        std::vector<char> engine_data(static_cast<size_t>(engine_size));
        if (!input.read(engine_data.data(), engine_size)) {
            throw std::runtime_error(
                "Failed to read ReID TensorRT engine file: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
        TRT_CHECK_THROW(runtime_ != nullptr,
                        "Failed to create ReID TensorRT runtime");

        engine_.reset(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        TRT_CHECK_THROW(engine_ != nullptr,
                        "Failed to deserialize ReID TensorRT engine");

        InspectBindings();
    }

    int bindingCount() const { return ::bindingCount(*engine_); }
    int inputBindingIndex() const { return input_binding_index_; }
    int outputBindingIndex() const { return output_binding_index_; }
    size_t inputBufferSize() const { return input_size_; }
    size_t outputBufferSize() const { return output_size_; }

    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>>
    createExecutionContext() const {
        auto execution_context =
            std::unique_ptr<nvinfer1::IExecutionContext,
                            TrtDeleter<nvinfer1::IExecutionContext>>(
                engine_->createExecutionContext());
        TRT_CHECK_THROW(execution_context != nullptr,
                        "Failed to create ReID TensorRT execution context");
        return execution_context;
    }

private:
    void InspectBindings() {
        const int total_bindings = ::bindingCount(*engine_);
        for (int binding_index = 0; binding_index < total_bindings; ++binding_index) {
            const size_t tensor_bytes = bindingSize(*engine_, binding_index);
            if (bindingIsInput(*engine_, binding_index)) {
                input_binding_index_ = binding_index;
                input_size_ = tensor_bytes;
                continue;
            }

            output_binding_index_ = binding_index;
            output_size_ = tensor_bytes;
        }

        if (input_binding_index_ < 0 || output_binding_index_ < 0) {
            throw std::runtime_error(
                "ReID TensorRT engine does not expose the expected bindings");
        }
    }

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    int input_binding_index_ = -1;
    int output_binding_index_ = -1;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
};

class ReidContext::ExecutionContextHandle {
public:
    explicit ExecutionContextHandle(
        std::unique_ptr<nvinfer1::IExecutionContext,
                        TrtDeleter<nvinfer1::IExecutionContext>> context)
        : context_(std::move(context)) {}

    nvinfer1::IExecutionContext& get() { return *context_; }

private:
    std::unique_ptr<nvinfer1::IExecutionContext,
                    TrtDeleter<nvinfer1::IExecutionContext>> context_;
};

ReidContext::ReidContext(
    std::shared_ptr<SharedReidModel> shared_model,
    int device_id)
    : shared_model_(std::move(shared_model)),
      device_id_(device_id),
      input_size_(0),
      output_size_(0) {
    try {
        bindCurrentThread();
        cudaStream_t stream = nullptr;
        CUDA_CHECK_THROW(cudaStreamCreate(&stream), "Failed to create CUDA stream");
        stream_ = stream;
        createExecutionContext();
        allocateBuffers();
    } catch (...) {
        releaseBuffers();
        if (stream_) {
            cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
            stream_ = nullptr;
        }
        throw;
    }
}

ReidContext::~ReidContext() {
    releaseBuffers();
    if (stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
}

void ReidContext::bindCurrentThread() {
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to bind CUDA device");
}

void ReidContext::copyToDevice(const void* data, size_t size) {
    if (size != input_size_) {
        throw std::runtime_error("ReID input size does not match TensorRT engine");
    }
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            device_input_ptr_,
            data,
            size,
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream_)),
        "Failed to copy ReID input tensor to device");
}

void ReidContext::execute() {
    TRT_CHECK_THROW(
        enqueueContext(
            execution_context_->get(),
            bindings_,
            static_cast<cudaStream_t>(stream_)),
        "ReID TensorRT inference failed");
}

void ReidContext::copyToHost(float* output_data, size_t element_count) {
    const size_t feature_bytes = element_count * sizeof(float);
    if (feature_bytes > output_size_) {
        throw std::runtime_error("Requested ReID feature size exceeds TensorRT output");
    }

    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            host_output_ptr_,
            device_output_ptr_,
            output_size_,
            cudaMemcpyDeviceToHost,
            static_cast<cudaStream_t>(stream_)),
        "Failed to copy ReID output tensor to host");
    CUDA_CHECK_THROW(
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)),
        "Failed to synchronize ReID CUDA stream");

    std::memcpy(output_data, host_output_ptr_, feature_bytes);
}

void ReidContext::createExecutionContext() {
    execution_context_ = std::make_unique<ExecutionContextHandle>(
        shared_model_->createExecutionContext());
}

void ReidContext::allocateBuffers() {
    const int binding_count = shared_model_->bindingCount();
    bindings_.assign(binding_count, nullptr);

    input_binding_index_ = shared_model_->inputBindingIndex();
    output_binding_index_ = shared_model_->outputBindingIndex();
    input_size_ = shared_model_->inputBufferSize();
    output_size_ = shared_model_->outputBufferSize();

    CUDA_CHECK_THROW(
        cudaMalloc(&device_input_ptr_, input_size_),
        "Failed to allocate ReID input buffer");
    CUDA_CHECK_THROW(
        cudaMalloc(&device_output_ptr_, output_size_),
        "Failed to allocate ReID output device buffer");
    CUDA_CHECK_THROW(
        cudaMallocHost(&host_output_ptr_, output_size_),
        "Failed to allocate ReID output host buffer");
    bindings_[input_binding_index_] = device_input_ptr_;
    bindings_[output_binding_index_] = device_output_ptr_;

    if (input_binding_index_ < 0 || output_binding_index_ < 0) {
        throw std::runtime_error(
            "ReID TensorRT engine bindings do not match deepsort_track_orin");
    }
}

void ReidContext::releaseBuffers() {
    if (device_input_ptr_) {
        cudaFree(device_input_ptr_);
        device_input_ptr_ = nullptr;
    }
    if (device_output_ptr_) {
        cudaFree(device_output_ptr_);
        device_output_ptr_ = nullptr;
    }
    if (host_output_ptr_) {
        cudaFreeHost(host_output_ptr_);
        host_output_ptr_ = nullptr;
    }

    bindings_.clear();
    input_binding_index_ = -1;
    output_binding_index_ = -1;
    input_size_ = 0;
    output_size_ = 0;
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateReidInferContexts(
    const std::string& engine_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("ReID context instance count must be greater than zero");
    }

    auto shared_model = std::make_shared<SharedReidModel>(engine_path);
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        contexts.push_back(std::make_shared<ReidContext>(shared_model, device_id));
    }
    return contexts;
}
