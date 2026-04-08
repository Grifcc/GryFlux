#include "infercontext.h"
#include "packet/fusion_data_packet.h"
#include "utils/logger.h"
#include <cuda_fp16.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#define CUDA_CHECK_BOOL(op, error_msg) do { \
    cudaError_t __ret = (op); \
    if (__ret != cudaSuccess) { \
        std::cerr << error_msg << ", ret=" << __ret << std::endl; \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_THROW(op, error_msg) do { \
    cudaError_t __ret = (op); \
    if (__ret != cudaSuccess) { \
        throw std::runtime_error(std::string(error_msg) + " failed! ret=" + std::to_string(__ret)); \
    } \
} while(0)

#define TRT_CHECK_THROW(op, error_msg) do { \
    if (!(op)) { \
        throw std::runtime_error(error_msg); \
    } \
} while(0)

namespace {

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
            throw std::runtime_error("Dynamic TensorRT shapes are not supported by this fusion sample");
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

size_t tensorSize(const nvinfer1::Dims& dims, nvinfer1::DataType data_type) {
    return volume(dims) * elementSize(data_type);
}

size_t bindingElementCount(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return volume(engine.getTensorShape(engine.getIOTensorName(binding_index)));
#else
    return volume(engine.getBindingDimensions(binding_index));
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

std::string bindingName(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getIOTensorName(binding_index);
#else
    return engine.getBindingName(binding_index);
#endif
}

nvinfer1::DataType bindingDataType(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorDataType(engine.getIOTensorName(binding_index));
#else
    return engine.getBindingDataType(binding_index);
#endif
}

nvinfer1::Dims bindingDims(const nvinfer1::ICudaEngine& engine, int binding_index) {
#if NV_TENSORRT_MAJOR >= 10
    return engine.getTensorShape(engine.getIOTensorName(binding_index));
#else
    return engine.getBindingDimensions(binding_index);
#endif
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    if (dims.nbDims < 0) {
        return "invalid";
    }
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        if (i > 0) {
            oss << "x";
        }
        oss << dims.d[i];
    }
    oss << "]";
    return oss.str();
}

const char* dataTypeToString(nvinfer1::DataType data_type) {
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

void convertFloatToBindingBuffer(const float* src, InferContext::TensorBinding& binding) {
    if (binding.dataType == nvinfer1::DataType::kFLOAT) {
        binding.hostBuffer.resize(binding.byteSize);
        std::memcpy(binding.hostBuffer.data(), src, binding.byteSize);
        return;
    }

    if (binding.dataType == nvinfer1::DataType::kHALF) {
        binding.hostBuffer.resize(binding.elementCount * sizeof(__half));
        auto* dst = reinterpret_cast<__half*>(binding.hostBuffer.data());
        for (size_t i = 0; i < binding.elementCount; ++i) {
            dst[i] = __float2half(src[i]);
        }
        return;
    }

    throw std::runtime_error("Fusion 输入仅支持 FP32/FP16 TensorRT tensor");
}

void convertBindingBufferToFloat(const InferContext::TensorBinding& binding, float* dst) {
    if (binding.dataType == nvinfer1::DataType::kFLOAT) {
        std::memcpy(dst, binding.hostBuffer.data(), binding.byteSize);
        return;
    }

    if (binding.dataType == nvinfer1::DataType::kHALF) {
        const auto* src = reinterpret_cast<const __half*>(binding.hostBuffer.data());
        for (size_t i = 0; i < binding.elementCount; ++i) {
            dst[i] = __half2float(src[i]);
        }
        return;
    }

    throw std::runtime_error("Fusion 输出仅支持 FP32/FP16 TensorRT tensor");
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

bool extractSpatialSize(const nvinfer1::Dims& dims, int& height, int& width) {
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

bool applySpatialSizeToDims(const nvinfer1::Dims& template_dims,
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

bool isDegenerateSpatialDims(const nvinfer1::Dims& dims) {
    int height = 0;
    int width = 0;
    return extractSpatialSize(dims, height, width) && height == 1 && width == 1;
}

bool hasWildcardDim(const nvinfer1::Dims& dims) {
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) {
            return true;
        }
    }
    return false;
}

#if NV_TENSORRT_MAJOR >= 10
bool tryGetProfileDims(const nvinfer1::ICudaEngine& engine,
                       const std::string& tensor_name,
                       int profile_index,
                       nvinfer1::OptProfileSelector selector,
                       nvinfer1::Dims& dims) {
    dims = engine.getProfileShape(tensor_name.c_str(), profile_index, selector);
    return dims.nbDims >= 0;
}

bool isUsableDataTensorDims(const nvinfer1::Dims& dims) {
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
#endif

}  // namespace

template <typename T>
void InferContext::TrtDeleter<T>::operator()(T* ptr) const {
    if (!ptr) {
        return;
    }
#if NV_TENSORRT_MAJOR >= 10
    delete ptr;
#else
    ptr->destroy();
#endif
}

InferContext::InferContext() 
    : deviceId_(0) {}

InferContext::~InferContext() {
    Destroy();
}

bool InferContext::Init(const std::string& modelPath, int deviceId) {
    try {
        Destroy();
        deviceId_ = deviceId;
        bindCurrentThread();
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_), "创建 CUDA Stream 失败");
        loadEngine(modelPath);
        return allocateBuffers();
    } catch (const std::exception& e) {
        std::cerr << "[InferContext] " << e.what() << std::endl;
        Destroy();
        return false;
    }
}

void InferContext::Destroy() {
    for (auto& binding : inputBindings_) {
        if (binding.devicePtr) {
            cudaFree(binding.devicePtr);
            binding.devicePtr = nullptr;
        }
    }
    inputBindings_.clear();
    bindings_.clear();

    if (outputBinding_.devicePtr) {
        cudaFree(outputBinding_.devicePtr);
        outputBinding_.devicePtr = nullptr;
    }
    outputBinding_ = TensorBinding{};

    context_.reset();
    engine_.reset();
    runtime_.reset();

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void InferContext::bindCurrentThread() {
    CUDA_CHECK_THROW(cudaSetDevice(deviceId_), "绑定 CUDA 设备失败");
}

void InferContext::copyInputToDevice(size_t index, const float* hostData, size_t elementCount) {
    if (index >= inputBindings_.size()) {
        throw std::runtime_error("Fusion 输入索引越界");
    }
    auto& binding = inputBindings_[index];
    if (elementCount != binding.elementCount) {
        throw std::runtime_error("Fusion 输入元素数量与 TensorRT Engine 不匹配");
    }
    convertFloatToBindingBuffer(hostData, binding);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(binding.devicePtr,
                        binding.hostBuffer.data(),
                        binding.byteSize,
                        cudaMemcpyHostToDevice,
                        stream_),
        "Fusion Host to Device 失败");
}

void InferContext::execute() {
    TRT_CHECK_THROW(enqueueContext(*context_, bindings_, stream_), "Fusion TensorRT 推理失败");
}

void InferContext::copyOutputToHost(float* hostData, size_t elementCount) {
    if (elementCount != outputBinding_.elementCount) {
        throw std::runtime_error("Fusion 输出元素数量与 TensorRT Engine 不匹配");
    }
    if (outputBinding_.dataType == nvinfer1::DataType::kFLOAT) {
        CUDA_CHECK_THROW(
            cudaMemcpyAsync(hostData,
                            outputBinding_.devicePtr,
                            outputBinding_.byteSize,
                            cudaMemcpyDeviceToHost,
                            stream_),
            "Fusion Device to Host 失败");
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream_), "同步 Fusion CUDA Stream 失败");
        return;
    }

    outputBinding_.hostBuffer.resize(outputBinding_.byteSize);
    CUDA_CHECK_THROW(
        cudaMemcpyAsync(outputBinding_.hostBuffer.data(),
                        outputBinding_.devicePtr,
                        outputBinding_.byteSize,
                        cudaMemcpyDeviceToHost,
                        stream_),
        "Fusion Device to Host 失败");
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_), "同步 Fusion CUDA Stream 失败");
    convertBindingBufferToFloat(outputBinding_, hostData);
}

void InferContext::loadEngine(const std::string& modelPath) {
    std::ifstream input(modelPath, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开 Fusion TensorRT Engine 文件: " + modelPath);
    }

    input.seekg(0, std::ios::end);
    const std::streamsize engine_size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (engine_size <= 0) {
        throw std::runtime_error("Fusion TensorRT Engine 文件为空: " + modelPath);
    }

    std::vector<char> engine_data(static_cast<size_t>(engine_size));
    if (!input.read(engine_data.data(), engine_size)) {
        throw std::runtime_error("读取 Fusion TensorRT Engine 文件失败: " + modelPath);
    }

    runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
    TRT_CHECK_THROW(runtime_ != nullptr, "创建 Fusion TensorRT Runtime 失败");

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    TRT_CHECK_THROW(engine_ != nullptr, "反序列化 Fusion TensorRT Engine 失败");

    context_.reset(engine_->createExecutionContext());
    TRT_CHECK_THROW(context_ != nullptr, "创建 Fusion TensorRT ExecutionContext 失败");
}

bool InferContext::allocateBuffers() {
    const int binding_count = bindingCount(*engine_);
    bindings_.assign(binding_count, nullptr);

#if NV_TENSORRT_MAJOR >= 10
    if (engine_->getNbOptimizationProfiles() > 0) {
        TRT_CHECK_THROW(context_->setOptimizationProfileAsync(0, stream_), "设置 Fusion TensorRT profile 失败");
    }
#endif

    for (int binding_index = 0; binding_index < binding_count; ++binding_index) {
        TensorBinding binding;
        binding.bindingIndex = binding_index;
        binding.name = bindingName(*engine_, binding_index);
        binding.dataType = bindingDataType(*engine_, binding_index);

        if (bindingIsInput(*engine_, binding_index)) {
#if NV_TENSORRT_MAJOR >= 10
            nvinfer1::Dims selected_dims = bindingDims(*engine_, binding_index);
            if (engine_->getNbOptimizationProfiles() > 0) {
                nvinfer1::Dims profile_dims;
                if (tryGetProfileDims(*engine_, binding.name, 0, nvinfer1::OptProfileSelector::kOPT, profile_dims)
                    && isUsableDataTensorDims(profile_dims)) {
                    selected_dims = profile_dims;
                }
            }

            if (!isUsableDataTensorDims(selected_dims) || isDegenerateSpatialDims(selected_dims)) {
                nvinfer1::Dims fallback_dims;
                if (applySpatialSizeToDims(bindingDims(*engine_, binding_index),
                                           GetFusionModelHeight(),
                                           GetFusionModelWidth(),
                                           fallback_dims)
                    && isUsableDataTensorDims(fallback_dims)) {
                    selected_dims = fallback_dims;
                }
            }

            TRT_CHECK_THROW(isUsableDataTensorDims(selected_dims), "Fusion 输入 tensor shape 非法");
            TRT_CHECK_THROW(context_->setInputShape(binding.name.c_str(), selected_dims), "设置 Fusion 输入 tensor shape 失败");
            binding.byteSize = tensorSize(selected_dims, binding.dataType);
            binding.elementCount = volume(selected_dims);
#else
            binding.byteSize = bindingSize(*engine_, binding_index);
            binding.elementCount = bindingElementCount(*engine_, binding_index);
#endif
            void* input_ptr = nullptr;
            CUDA_CHECK_BOOL(cudaMalloc(&input_ptr, binding.byteSize), "分配 Fusion 输入显存失败");
            binding.devicePtr = input_ptr;
            binding.hostBuffer.resize(binding.byteSize);
            inputBindings_.push_back(std::move(binding));
            bindings_[binding_index] = input_ptr;
            continue;
        }

        if (outputBinding_.bindingIndex >= 0) {
            std::cerr << "[InferContext] Fusion Engine 只支持 1 个输出 tensor" << std::endl;
            return false;
        }

#if NV_TENSORRT_MAJOR >= 10
        binding.byteSize = 0;
        binding.elementCount = 0;
#else
        binding.byteSize = bindingSize(*engine_, binding_index);
        binding.elementCount = bindingElementCount(*engine_, binding_index);
#endif
        if (binding.byteSize > 0) {
            CUDA_CHECK_BOOL(cudaMalloc(&binding.devicePtr, binding.byteSize), "分配 Fusion 输出显存失败");
            binding.hostBuffer.resize(binding.byteSize);
        }
        bindings_[binding_index] = binding.devicePtr;
        outputBinding_ = std::move(binding);
    }

    if (inputBindings_.size() != 2 || outputBinding_.bindingIndex < 0) {
        std::cerr << "[InferContext] Fusion Engine 必须包含 2 个输入和 1 个输出" << std::endl;
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    const int infer_status = context_->inferShapes(0, nullptr);
    TRT_CHECK_THROW(infer_status == 0, "Fusion TensorRT inferShapes 失败");

    for (auto& binding : inputBindings_) {
        const auto actual_dims = context_->getTensorShape(binding.name.c_str());
        TRT_CHECK_THROW(isUsableDataTensorDims(actual_dims), "Fusion 输入实际 shape 非法");
        binding.byteSize = tensorSize(actual_dims, binding.dataType);
        binding.elementCount = volume(actual_dims);
        binding.hostBuffer.resize(binding.byteSize);
    }

    const auto output_dims = context_->getTensorShape(outputBinding_.name.c_str());
    TRT_CHECK_THROW(isUsableDataTensorDims(output_dims), "Fusion 输出实际 shape 非法");
    outputBinding_.byteSize = tensorSize(output_dims, outputBinding_.dataType);
    outputBinding_.elementCount = volume(output_dims);
    cudaFree(outputBinding_.devicePtr);
    outputBinding_.devicePtr = nullptr;
    CUDA_CHECK_BOOL(cudaMalloc(&outputBinding_.devicePtr, outputBinding_.byteSize), "分配 Fusion 输出显存失败");
    outputBinding_.hostBuffer.resize(outputBinding_.byteSize);
    bindings_[outputBinding_.bindingIndex] = outputBinding_.devicePtr;
    const auto spatial_dims = context_->getTensorShape(inputBindings_.front().name.c_str());
#else
    const auto spatial_dims = bindingDims(*engine_, inputBindings_.front().bindingIndex);
#endif

    int model_height = 0;
    int model_width = 0;
    if (!extractSpatialSize(spatial_dims, model_height, model_width)) {
        std::cerr << "[InferContext] 无法从 Fusion 输入 tensor shape 推断图像尺寸" << std::endl;
        return false;
    }
    SetFusionModelSize(model_width, model_height);

    logBindings();
    return true;
}

void InferContext::logBindings() const {
    LOG.info("Fusion TensorRT inputs:");
    for (const auto& binding : inputBindings_) {
        const auto dims =
#if NV_TENSORRT_MAJOR >= 10
            context_->getTensorShape(binding.name.c_str());
#else
            bindingDims(*engine_, binding.bindingIndex);
#endif
        LOG.info("  [%d] name=%s dtype=%s dims=%s bytes=%zu elements=%zu",
                 binding.bindingIndex,
                 binding.name.c_str(),
                 dataTypeToString(binding.dataType),
                 dimsToString(dims).c_str(),
                 binding.byteSize,
                 binding.elementCount);
    }

    const auto output_dims =
#if NV_TENSORRT_MAJOR >= 10
        context_->getTensorShape(outputBinding_.name.c_str());
#else
        bindingDims(*engine_, outputBinding_.bindingIndex);
#endif
    LOG.info("Fusion TensorRT output:");
    LOG.info("  [%d] name=%s dtype=%s dims=%s bytes=%zu elements=%zu",
             outputBinding_.bindingIndex,
             outputBinding_.name.c_str(),
             dataTypeToString(outputBinding_.dataType),
             dimsToString(output_dims).c_str(),
             outputBinding_.byteSize,
             outputBinding_.elementCount);
}
