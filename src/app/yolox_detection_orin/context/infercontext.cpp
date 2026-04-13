#include "infercontext.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

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
            throw std::runtime_error("Dynamic TensorRT shapes are not supported by this Orin sample");
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

class SharedInferModel {
public:
    explicit SharedInferModel(const std::string& engine_model_path) {
        std::ifstream input(engine_model_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Failed to open TensorRT engine file: " + engine_model_path);
        }

        input.seekg(0, std::ios::end);
        const std::streamsize engine_size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (engine_size <= 0) {
            throw std::runtime_error(
                "TensorRT engine file is empty: " + engine_model_path);
        }

        std::vector<char> engine_data(static_cast<size_t>(engine_size));
        if (!input.read(engine_data.data(), engine_size)) {
            throw std::runtime_error(
                "Failed to read TensorRT engine file: " + engine_model_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
        TRT_CHECK_THROW(runtime_ != nullptr, "Failed to create TensorRT runtime");

        engine_.reset(
            runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        TRT_CHECK_THROW(engine_ != nullptr, "Failed to deserialize TensorRT engine");

        InspectBindings();
    }

    int bindingCount() const { return ::bindingCount(*engine_); }

    int inputBindingIndex() const { return input_binding_index_; }

    size_t inputBufferSize() const { return input_buffer_size_; }

    const std::vector<int>& outputBindingIndices() const {
        return output_binding_indices_;
    }

    const std::vector<size_t>& outputBufferSizes() const {
        return output_buffer_sizes_;
    }

    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>>
    createExecutionContext() const {
        auto execution_context =
            std::unique_ptr<nvinfer1::IExecutionContext,
                            TrtDeleter<nvinfer1::IExecutionContext>>(
                engine_->createExecutionContext());
        TRT_CHECK_THROW(execution_context != nullptr,
                        "Failed to create TensorRT execution context");
        return execution_context;
    }

private:
    void InspectBindings() {
        const int total_bindings = ::bindingCount(*engine_);
        for (int binding_index = 0; binding_index < total_bindings; ++binding_index) {
            const size_t tensor_bytes = bindingSize(*engine_, binding_index);
            if (bindingIsInput(*engine_, binding_index)) {
                input_binding_index_ = binding_index;
                input_buffer_size_ = tensor_bytes;
                continue;
            }
            output_binding_indices_.push_back(binding_index);
            output_buffer_sizes_.push_back(tensor_bytes);
        }

        if (input_binding_index_ < 0 || output_binding_indices_.empty()) {
            throw std::runtime_error(
                "TensorRT engine does not expose the expected input/output bindings");
        }
    }

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    int input_binding_index_ = -1;
    size_t input_buffer_size_ = 0;
    std::vector<int> output_binding_indices_;
    std::vector<size_t> output_buffer_sizes_;
};

class InferContext::ExecutionContextHandle {
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

InferContext::InferContext(
    std::shared_ptr<SharedInferModel> shared_model,
    int device_id)
    : shared_model_(std::move(shared_model)),
      device_id_(device_id) {
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

InferContext::~InferContext() {
    releaseBuffers();
    if (stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
        stream_ = nullptr;
    }
}

void InferContext::bindCurrentThread() {
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to bind CUDA device");
}

void InferContext::copyToDevice(const void* host_data, size_t size) {
    if (size != input_buffer_size_) {
        throw std::runtime_error("YOLOX input size does not match TensorRT engine");
    }
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            input_buffer_,
            host_data,
            size,
            cudaMemcpyHostToDevice,
            static_cast<cudaStream_t>(stream_)),
        "Failed to copy input tensor to device");
}

void InferContext::executeModel() {
    TRT_CHECK_THROW(
        enqueueContext(
            execution_context_->get(),
            bindings_,
            static_cast<cudaStream_t>(stream_)),
        "TensorRT inference failed");
}

void InferContext::copyToHost() {
    for (auto& out_buf : output_buffers_) {
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                out_buf.host_buffer,
                out_buf.device_buffer,
                out_buf.size,
                cudaMemcpyDeviceToHost,
                static_cast<cudaStream_t>(stream_)),
            "Failed to copy output tensor to host");
    }
    CUDA_CHECK_THROW(
        cudaStreamSynchronize(static_cast<cudaStream_t>(stream_)),
        "Failed to synchronize CUDA stream");
}

void InferContext::createExecutionContext() {
    execution_context_ = std::make_unique<ExecutionContextHandle>(
        shared_model_->createExecutionContext());
}

void InferContext::allocateBuffers() {
    const int binding_count = shared_model_->bindingCount();
    bindings_.assign(binding_count, nullptr);

    input_index_ = shared_model_->inputBindingIndex();
    input_buffer_size_ = shared_model_->inputBufferSize();
    CUDA_CHECK_THROW(
        cudaMalloc(&input_buffer_, input_buffer_size_),
        "Failed to allocate TensorRT input buffer");
    bindings_[input_index_] = input_buffer_;

    const auto& output_binding_indices = shared_model_->outputBindingIndices();
    const auto& output_buffer_sizes = shared_model_->outputBufferSizes();
    output_indices_ = output_binding_indices;
    output_buffers_.reserve(output_binding_indices.size());

    for (size_t index = 0; index < output_binding_indices.size(); ++index) {
        ModelOutput output;
        output.size = output_buffer_sizes[index];
        CUDA_CHECK_THROW(
            cudaMalloc(&output.device_buffer, output.size),
            "Failed to allocate TensorRT output device buffer");
        CUDA_CHECK_THROW(
            cudaMallocHost(&output.host_buffer, output.size),
            "Failed to allocate TensorRT output host buffer");
        bindings_[output_binding_indices[index]] = output.device_buffer;
        output_buffers_.push_back(output);
    }

    if (input_index_ < 0 || output_buffers_.empty()) {
        throw std::runtime_error(
            "TensorRT engine bindings do not match yolox_detection_orin");
    }
}

void InferContext::releaseBuffers() {
    if (input_buffer_) {
        cudaFree(input_buffer_);
        input_buffer_ = nullptr;
    }

    for (auto& out : output_buffers_) {
        if (out.device_buffer) {
            cudaFree(out.device_buffer);
            out.device_buffer = nullptr;
        }
        if (out.host_buffer) {
            cudaFreeHost(out.host_buffer);
            out.host_buffer = nullptr;
        }
    }

    output_buffers_.clear();
    output_indices_.clear();
    bindings_.clear();
    input_buffer_size_ = 0;
    input_index_ = -1;
}

std::vector<std::shared_ptr<GryFlux::Context>> CreateInferContexts(
    const std::string& engine_model_path,
    int device_id,
    size_t instance_count) {
    if (instance_count == 0) {
        throw std::runtime_error("TensorRT context instance count must be greater than zero");
    }

    auto shared_model = std::make_shared<SharedInferModel>(engine_model_path);
    std::vector<std::shared_ptr<GryFlux::Context>> contexts;
    contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        contexts.push_back(std::make_shared<InferContext>(shared_model, device_id));
    }
    return contexts;
}
