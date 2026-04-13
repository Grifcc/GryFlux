#include "context/infercontext.h"

#include "utils/logger.h"

#include <cuda_fp16.h>

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
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

size_t ElementSize(nvinfer1::DataType data_type) {
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

size_t Volume(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            throw std::runtime_error("Dynamic TensorRT dims were not fully resolved");
        }
        size *= static_cast<size_t>(dims.d[i]);
    }
    return size;
}

size_t TensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType data_type) {
    return Volume(dims) * ElementSize(data_type);
}

int BindingCount(const nvinfer1::ICudaEngine& engine) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getNbIOTensors();
#else
    return engine.getNbBindings();
#endif
}

bool BindingIsInput(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorIOMode(engine.getIOTensorName(binding_index)) ==
           nvinfer1::TensorIOMode::kINPUT;
#else
    return engine.bindingIsInput(binding_index);
#endif
}

std::string BindingName(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getIOTensorName(binding_index);
#else
    return engine.getBindingName(binding_index);
#endif
}

nvinfer1::DataType BindingDataType(
    const nvinfer1::ICudaEngine& engine,
    int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorDataType(engine.getIOTensorName(binding_index));
#else
    return engine.getBindingDataType(binding_index);
#endif
}

nvinfer1::Dims BindingDims(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorShape(engine.getIOTensorName(binding_index));
#else
    return engine.getBindingDimensions(binding_index);
#endif
}

std::string DimsToString(const nvinfer1::Dims& dims) {
    std::ostringstream output;
    output << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            output << "x";
        }
        output << dims.d[i];
    }
    output << "]";
    return output.str();
}

const char* DataTypeToString(nvinfer1::DataType data_type) {
    switch (data_type) {
        case nvinfer1::DataType::kFLOAT:
            return "FP32";
        case nvinfer1::DataType::kHALF:
            return "FP16";
        case nvinfer1::DataType::kINT8:
            return "INT8";
        case nvinfer1::DataType::kINT32:
            return "INT32";
        case nvinfer1::DataType::kBOOL:
            return "BOOL";
        case nvinfer1::DataType::kUINT8:
            return "UINT8";
#if NV_TENSORRT_MAJOR >= 8
        case nvinfer1::DataType::kINT64:
            return "INT64";
#endif
        default:
            return "UNKNOWN";
    }
}

bool EnqueueContext(
    nvinfer1::IExecutionContext& context,
    std::vector<void*>& bindings,
    cudaStream_t stream) {
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

bool ExtractSpatialSize(const nvinfer1::Dims& dims, int& height, int& width) {
    if (dims.nbDims == 2) {
        height = dims.d[0];
        width = dims.d[1];
        return height > 0 && width > 0;
    }
    if (dims.nbDims == 3) {
        if (dims.d[0] == 1) {
            height = dims.d[1];
            width = dims.d[2];
            return height > 0 && width > 0;
        }
        if (dims.d[2] == 1) {
            height = dims.d[0];
            width = dims.d[1];
            return height > 0 && width > 0;
        }
    }
    if (dims.nbDims == 4) {
        if (dims.d[1] == 1) {
            height = dims.d[2];
            width = dims.d[3];
            return height > 0 && width > 0;
        }
        if (dims.d[3] == 1) {
            height = dims.d[1];
            width = dims.d[2];
            return height > 0 && width > 0;
        }
    }
    return false;
}

bool ApplySpatialSizeToDims(
    const nvinfer1::Dims& template_dims,
    int height,
    int width,
    nvinfer1::Dims& resolved_dims) {
    resolved_dims = template_dims;
    if (resolved_dims.nbDims == 2) {
        resolved_dims.d[0] = height;
        resolved_dims.d[1] = width;
        return true;
    }
    if (resolved_dims.nbDims == 3) {
        if (resolved_dims.d[0] == 1) {
            resolved_dims.d[1] = height;
            resolved_dims.d[2] = width;
            return true;
        }
        if (resolved_dims.d[2] == 1) {
            resolved_dims.d[0] = height;
            resolved_dims.d[1] = width;
            return true;
        }
    }
    if (resolved_dims.nbDims == 4) {
        if (resolved_dims.d[1] == 1) {
            resolved_dims.d[0] = 1;
            resolved_dims.d[1] = 1;
            resolved_dims.d[2] = height;
            resolved_dims.d[3] = width;
            return true;
        }
        if (resolved_dims.d[3] == 1) {
            resolved_dims.d[0] = 1;
            resolved_dims.d[1] = height;
            resolved_dims.d[2] = width;
            resolved_dims.d[3] = 1;
            return true;
        }
    }
    return false;
}

bool IsDegenerateSpatialDims(const nvinfer1::Dims& dims) {
    int height = 0;
    int width = 0;
    return ExtractSpatialSize(dims, height, width) && height == 1 && width == 1;
}

bool IsUsableDataTensorDims(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        return false;
    }
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) {
            return false;
        }
    }
    return true;
}

#if NV_TENSORRT_MAJOR >= 10
bool TryGetProfileDims(
    const nvinfer1::ICudaEngine& engine,
    const std::string& tensor_name,
    int profile_index,
    nvinfer1::OptProfileSelector selector,
    nvinfer1::Dims& dims) {
    dims = engine.getProfileShape(tensor_name.c_str(), profile_index, selector);
    return dims.nbDims >= 0;
}
#endif

void ConvertFloatToBindingBuffer(
    const float* src,
    InferContext::TensorBuffer& binding) {
    if (binding.data_type == nvinfer1::DataType::kFLOAT) {
        binding.host_buffer.resize(binding.byte_size);
        std::memcpy(binding.host_buffer.data(), src, binding.byte_size);
        return;
    }

    if (binding.data_type == nvinfer1::DataType::kHALF) {
        binding.host_buffer.resize(binding.element_count * sizeof(__half));
        auto* dst = reinterpret_cast<__half*>(binding.host_buffer.data());
        for (size_t index = 0; index < binding.element_count; ++index) {
            dst[index] = __float2half(src[index]);
        }
        return;
    }

    throw std::runtime_error("Fusion input tensor only supports FP32/FP16");
}

void ConvertBindingBufferToFloat(
    const InferContext::TensorBuffer& binding,
    float* dst) {
    if (binding.data_type == nvinfer1::DataType::kFLOAT) {
        std::memcpy(dst, binding.host_buffer.data(), binding.byte_size);
        return;
    }

    if (binding.data_type == nvinfer1::DataType::kHALF) {
        const auto* src =
            reinterpret_cast<const __half*>(binding.host_buffer.data());
        for (size_t index = 0; index < binding.element_count; ++index) {
            dst[index] = __half2float(src[index]);
        }
        return;
    }

    throw std::runtime_error("Fusion output tensor only supports FP32/FP16");
}

struct FusionBindingInfo {
    int binding_index = -1;
    std::string name;
    nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;
    nvinfer1::Dims dims{};
    size_t byte_size = 0;
    size_t element_count = 0;
};

}  // namespace

class SharedFusionModel {
public:
    SharedFusionModel(
        const std::string& engine_path,
        int device_id,
        int fallback_width,
        int fallback_height) {
        LoadEngine(engine_path);
        InspectBindings(device_id, fallback_width, fallback_height);
    }

    int bindingCount() const { return binding_count_; }

    const std::vector<FusionBindingInfo>& inputBindings() const {
        return input_bindings_;
    }

    const FusionBindingInfo& outputBinding() const {
        return output_binding_;
    }

    FusionModelInfo modelInfo() const {
        FusionModelInfo info;
        info.model_width = model_width_;
        info.model_height = model_height_;
        if (input_bindings_.size() >= 2) {
            info.vis_input_elements = input_bindings_[0].element_count;
            info.ir_input_elements = input_bindings_[1].element_count;
        }
        info.output_elements = output_binding_.element_count;
        return info;
    }

    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>>
    createExecutionContext(cudaStream_t stream) const {
        auto context =
            std::unique_ptr<nvinfer1::IExecutionContext,
                            TrtDeleter<nvinfer1::IExecutionContext>>(
                engine_->createExecutionContext());
        TRT_CHECK_THROW(context != nullptr, "Failed to create Fusion TensorRT context");

#if NV_TENSORRT_MAJOR >= 10
        if (engine_->getNbOptimizationProfiles() > 0) {
            TRT_CHECK_THROW(
                context->setOptimizationProfileAsync(0, stream),
                "Failed to set Fusion TensorRT profile");
        }
        for (const auto& binding : input_bindings_) {
            TRT_CHECK_THROW(
                context->setInputShape(binding.name.c_str(), binding.dims),
                "Failed to set Fusion TensorRT input shape");
        }
        TRT_CHECK_THROW(
            context->inferShapes(0, nullptr) == 0,
            "Fusion TensorRT inferShapes failed");
#endif
        return context;
    }

private:
    void LoadEngine(const std::string& engine_path) {
        std::ifstream input(engine_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "Failed to open Fusion TensorRT engine file: " + engine_path);
        }

        input.seekg(0, std::ios::end);
        const std::streamsize engine_size = input.tellg();
        input.seekg(0, std::ios::beg);
        if (engine_size <= 0) {
            throw std::runtime_error(
                "Fusion TensorRT engine file is empty: " + engine_path);
        }

        std::vector<char> engine_data(static_cast<size_t>(engine_size));
        if (!input.read(engine_data.data(), engine_size)) {
            throw std::runtime_error(
                "Failed to read Fusion TensorRT engine file: " + engine_path);
        }

        runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
        TRT_CHECK_THROW(runtime_ != nullptr, "Failed to create Fusion TensorRT runtime");

        engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        TRT_CHECK_THROW(engine_ != nullptr, "Failed to deserialize Fusion TensorRT engine");
        binding_count_ = BindingCount(*engine_);
    }

    void InspectBindings(int device_id, int fallback_width, int fallback_height) {
        CUDA_CHECK_THROW(cudaSetDevice(device_id), "Failed to bind CUDA device");
        cudaStream_t inspection_stream = nullptr;
        CUDA_CHECK_THROW(
            cudaStreamCreate(&inspection_stream),
            "Failed to create Fusion inspection CUDA stream");

        try {
            auto context = createExecutionContext(inspection_stream);
            for (int binding_index = 0; binding_index < binding_count_; ++binding_index) {
                FusionBindingInfo binding;
                binding.binding_index = binding_index;
                binding.name = BindingName(*engine_, binding_index);
                binding.data_type = BindingDataType(*engine_, binding_index);

                if (BindingIsInput(*engine_, binding_index)) {
#if NV_TENSORRT_MAJOR >= 10
                    nvinfer1::Dims selected_dims = BindingDims(*engine_, binding_index);
                    if (engine_->getNbOptimizationProfiles() > 0) {
                        nvinfer1::Dims profile_dims;
                        if (TryGetProfileDims(
                                *engine_,
                                binding.name,
                                0,
                                nvinfer1::OptProfileSelector::kOPT,
                                profile_dims) &&
                            IsUsableDataTensorDims(profile_dims)) {
                            selected_dims = profile_dims;
                        }
                    }
                    if ((!IsUsableDataTensorDims(selected_dims) ||
                         IsDegenerateSpatialDims(selected_dims)) &&
                        fallback_width > 0 && fallback_height > 0) {
                        nvinfer1::Dims fallback_dims;
                        if (ApplySpatialSizeToDims(
                                BindingDims(*engine_, binding_index),
                                fallback_height,
                                fallback_width,
                                fallback_dims) &&
                            IsUsableDataTensorDims(fallback_dims)) {
                            selected_dims = fallback_dims;
                        }
                    }
                    TRT_CHECK_THROW(
                        IsUsableDataTensorDims(selected_dims),
                        "Fusion input tensor shape is invalid");
                    binding.dims = selected_dims;
#else
                    binding.dims = BindingDims(*engine_, binding_index);
#endif
                    input_bindings_.push_back(binding);
                    continue;
                }

                if (output_binding_.binding_index >= 0) {
                    throw std::runtime_error(
                        "Fusion engine is expected to expose exactly one output tensor");
                }
                binding.dims = BindingDims(*engine_, binding_index);
                output_binding_ = binding;
            }

            TRT_CHECK_THROW(
                input_bindings_.size() == 2 && output_binding_.binding_index >= 0,
                "Fusion engine must expose 2 inputs and 1 output");

#if NV_TENSORRT_MAJOR >= 10
            for (const auto& binding : input_bindings_) {
                TRT_CHECK_THROW(
                    context->setInputShape(binding.name.c_str(), binding.dims),
                    "Failed to set Fusion input shape during inspection");
            }
            TRT_CHECK_THROW(
                context->inferShapes(0, nullptr) == 0,
                "Fusion TensorRT inferShapes failed during inspection");
            for (auto& binding : input_bindings_) {
                binding.dims = context->getTensorShape(binding.name.c_str());
            }
            output_binding_.dims = context->getTensorShape(output_binding_.name.c_str());
#endif

            for (auto& binding : input_bindings_) {
                TRT_CHECK_THROW(
                    IsUsableDataTensorDims(binding.dims),
                    "Fusion input dims remain unresolved");
                binding.byte_size = TensorSize(binding.dims, binding.data_type);
                binding.element_count = Volume(binding.dims);
            }

            TRT_CHECK_THROW(
                IsUsableDataTensorDims(output_binding_.dims),
                "Fusion output dims remain unresolved");
            output_binding_.byte_size =
                TensorSize(output_binding_.dims, output_binding_.data_type);
            output_binding_.element_count = Volume(output_binding_.dims);

            TRT_CHECK_THROW(
                ExtractSpatialSize(input_bindings_.front().dims, model_height_, model_width_),
                "Failed to infer Fusion model spatial size");

            LogBindings();
        } catch (...) {
            if (inspection_stream) {
                cudaStreamDestroy(inspection_stream);
            }
            throw;
        }

        if (inspection_stream) {
            cudaStreamDestroy(inspection_stream);
        }
    }

    void LogBindings() const {
        LOG.info("Fusion TensorRT inputs:");
        for (const auto& binding : input_bindings_) {
            LOG.info(
                "  [%d] name=%s dtype=%s dims=%s bytes=%zu elements=%zu",
                binding.binding_index,
                binding.name.c_str(),
                DataTypeToString(binding.data_type),
                DimsToString(binding.dims).c_str(),
                binding.byte_size,
                binding.element_count);
        }
        LOG.info("Fusion TensorRT output:");
        LOG.info(
            "  [%d] name=%s dtype=%s dims=%s bytes=%zu elements=%zu",
            output_binding_.binding_index,
            output_binding_.name.c_str(),
            DataTypeToString(output_binding_.data_type),
            DimsToString(output_binding_.dims).c_str(),
            output_binding_.byte_size,
            output_binding_.element_count);
    }

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    int binding_count_ = 0;
    std::vector<FusionBindingInfo> input_bindings_;
    FusionBindingInfo output_binding_;
    int model_width_ = 0;
    int model_height_ = 0;
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
    std::shared_ptr<SharedFusionModel> shared_model,
    int device_id)
    : shared_model_(std::move(shared_model)),
      device_id_(device_id) {
    try {
        bindCurrentThread();
        CUDA_CHECK_THROW(
            cudaStreamCreate(&stream_),
            "Failed to create Fusion CUDA stream");
        createExecutionContext();
        allocateBuffers();
    } catch (...) {
        releaseBuffers();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        throw;
    }
}

InferContext::~InferContext() {
    releaseBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void InferContext::bindCurrentThread() {
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "Failed to bind CUDA device");
}

size_t InferContext::GetInputElementCount(size_t index) const {
    return input_buffers_.at(index).element_count;
}

size_t InferContext::GetOutputElementCount() const {
    return output_buffer_.element_count;
}

void InferContext::copyInputToDevice(
    size_t index,
    const float* host_data,
    size_t element_count) {
    auto& binding = input_buffers_.at(index);
    if (element_count != binding.element_count) {
        throw std::runtime_error(
            "Fusion input element count does not match TensorRT engine");
    }

    ConvertFloatToBindingBuffer(host_data, binding);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            binding.device_ptr,
            binding.host_buffer.data(),
            binding.byte_size,
            cudaMemcpyHostToDevice,
            stream_),
        "Failed to copy Fusion input tensor to device");
}

void InferContext::execute() {
    TRT_CHECK_THROW(
        EnqueueContext(execution_context_->get(), bindings_, stream_),
        "Fusion TensorRT inference failed");
}

void InferContext::copyOutputToHost(float* host_data, size_t element_count) {
    if (element_count != output_buffer_.element_count) {
        throw std::runtime_error(
            "Fusion output element count does not match TensorRT engine");
    }

    if (output_buffer_.data_type == nvinfer1::DataType::kFLOAT) {
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(
                host_data,
                output_buffer_.device_ptr,
                output_buffer_.byte_size,
                cudaMemcpyDeviceToHost,
                stream_),
            "Failed to copy Fusion output tensor to host");
        CUDA_CHECK_THROW(
            cudaStreamSynchronize(stream_),
            "Failed to synchronize Fusion CUDA stream");
        return;
    }

    output_buffer_.host_buffer.resize(output_buffer_.byte_size);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(
            output_buffer_.host_buffer.data(),
            output_buffer_.device_ptr,
            output_buffer_.byte_size,
            cudaMemcpyDeviceToHost,
            stream_),
        "Failed to copy Fusion output tensor to host");
    CUDA_CHECK_THROW(
        cudaStreamSynchronize(stream_),
        "Failed to synchronize Fusion CUDA stream");
    ConvertBindingBufferToFloat(output_buffer_, host_data);
}

void InferContext::createExecutionContext() {
    execution_context_ = std::make_unique<ExecutionContextHandle>(
        shared_model_->createExecutionContext(stream_));
}

void InferContext::allocateBuffers() {
    bindings_.assign(static_cast<size_t>(shared_model_->bindingCount()), nullptr);
    input_buffers_.clear();

    for (const auto& binding_info : shared_model_->inputBindings()) {
        TensorBuffer buffer;
        buffer.binding_index = binding_info.binding_index;
        buffer.name = binding_info.name;
        buffer.data_type = binding_info.data_type;
        buffer.byte_size = binding_info.byte_size;
        buffer.element_count = binding_info.element_count;
        buffer.host_buffer.resize(buffer.byte_size);
        CUDA_CHECK_THROW(
            cudaMalloc(&buffer.device_ptr, buffer.byte_size),
            "Failed to allocate Fusion input buffer");
        bindings_[static_cast<size_t>(buffer.binding_index)] = buffer.device_ptr;
        input_buffers_.push_back(std::move(buffer));
    }

    const auto& output_binding = shared_model_->outputBinding();
    output_buffer_.binding_index = output_binding.binding_index;
    output_buffer_.name = output_binding.name;
    output_buffer_.data_type = output_binding.data_type;
    output_buffer_.byte_size = output_binding.byte_size;
    output_buffer_.element_count = output_binding.element_count;
    if (output_buffer_.data_type != nvinfer1::DataType::kFLOAT) {
        output_buffer_.host_buffer.resize(output_buffer_.byte_size);
    }
    CUDA_CHECK_THROW(
        cudaMalloc(&output_buffer_.device_ptr, output_buffer_.byte_size),
        "Failed to allocate Fusion output buffer");
    bindings_[static_cast<size_t>(output_buffer_.binding_index)] =
        output_buffer_.device_ptr;
}

void InferContext::releaseBuffers() {
    execution_context_.reset();

    for (auto& buffer : input_buffers_) {
        if (buffer.device_ptr) {
            cudaFree(buffer.device_ptr);
            buffer.device_ptr = nullptr;
        }
    }
    input_buffers_.clear();

    if (output_buffer_.device_ptr) {
        cudaFree(output_buffer_.device_ptr);
        output_buffer_.device_ptr = nullptr;
    }
    output_buffer_ = TensorBuffer{};
    bindings_.clear();
}

FusionInferResourceBundle CreateFusionInferResourceBundle(
    const std::string& engine_path,
    int device_id,
    size_t instance_count,
    int fallback_width,
    int fallback_height) {
    if (instance_count == 0) {
        throw std::runtime_error(
            "Fusion infer context instance count must be greater than zero");
    }

    auto shared_model = std::make_shared<SharedFusionModel>(
        engine_path,
        device_id,
        fallback_width,
        fallback_height);

    FusionInferResourceBundle bundle;
    bundle.model_info = shared_model->modelInfo();
    bundle.contexts.reserve(instance_count);
    for (size_t index = 0; index < instance_count; ++index) {
        bundle.contexts.push_back(
            std::make_shared<InferContext>(shared_model, device_id));
    }
    return bundle;
}
