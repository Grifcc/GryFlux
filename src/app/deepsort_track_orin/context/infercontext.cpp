#include "infercontext.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

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

InferContext::InferContext(const std::string& engine_model_path, int device_id)
    : device_id_(device_id) 
{
    try {
        bindCurrentThread();
        CUDA_CHECK_THROW(cudaStreamCreate(&stream_), "创建 CUDA Stream 失败");
        loadEngine(engine_model_path);
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
    CUDA_CHECK_THROW(cudaSetDevice(device_id_), "绑定 CUDA 设备失败");
}

void InferContext::copyToDevice(const void* host_data, size_t size) {
    if (size != input_buffer_size_) {
        throw std::runtime_error("DeepSORT 检测输入大小与 TensorRT Engine 不匹配");
    }
    CUDA_CHECK_THROW(cudaMemcpyAsync(input_buffer_, host_data, size, cudaMemcpyHostToDevice, stream_), "Host to Device 失败");
}

void InferContext::executeModel() {
    TRT_CHECK_THROW(enqueueContext(*context_, bindings_, stream_), "TensorRT 模型推理失败");
}

void InferContext::copyToHost() {
    for (auto& out_buf : output_buffers_) {
        CUDA_CHECK_THROW(cudaMemcpyAsync(out_buf.host_buffer,
                                         out_buf.device_buffer,
                                         out_buf.size,
                                         cudaMemcpyDeviceToHost,
                                         stream_),
                         "Device to Host 失败");
    }
    CUDA_CHECK_THROW(cudaStreamSynchronize(stream_), "同步 CUDA Stream 失败");
}

void InferContext::loadEngine(const std::string& engine_model_path) {
    std::ifstream input(engine_model_path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("无法打开 TensorRT Engine 文件: " + engine_model_path);
    }

    input.seekg(0, std::ios::end);
    const std::streamsize engine_size = input.tellg();
    input.seekg(0, std::ios::beg);

    if (engine_size <= 0) {
        throw std::runtime_error("TensorRT Engine 文件为空: " + engine_model_path);
    }

    std::vector<char> engine_data(static_cast<size_t>(engine_size));
    if (!input.read(engine_data.data(), engine_size)) {
        throw std::runtime_error("读取 TensorRT Engine 文件失败: " + engine_model_path);
    }

    runtime_.reset(nvinfer1::createInferRuntime(trtLogger()));
    TRT_CHECK_THROW(runtime_ != nullptr, "创建 TensorRT Runtime 失败");

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    TRT_CHECK_THROW(engine_ != nullptr, "反序列化 TensorRT Engine 失败");

    context_.reset(engine_->createExecutionContext());
    TRT_CHECK_THROW(context_ != nullptr, "创建 TensorRT ExecutionContext 失败");
}

void InferContext::allocateBuffers() {
    const int binding_count = bindingCount(*engine_);
    bindings_.assign(binding_count, nullptr);

    for (int binding_index = 0; binding_index < binding_count; ++binding_index) {
        const size_t tensor_bytes = bindingSize(*engine_, binding_index);
        if (bindingIsInput(*engine_, binding_index)) {
            input_index_ = binding_index;
            input_buffer_size_ = tensor_bytes;
            CUDA_CHECK_THROW(cudaMalloc(&input_buffer_, input_buffer_size_), "分配输入显存失败");
            bindings_[binding_index] = input_buffer_;
            continue;
        }

        ModelOutput output;
        output.size = tensor_bytes;
        CUDA_CHECK_THROW(cudaMalloc(&output.device_buffer, output.size), "分配输出显存失败");
        CUDA_CHECK_THROW(cudaMallocHost(&output.host_buffer, output.size), "分配输出主机内存失败");
        bindings_[binding_index] = output.device_buffer;
        output_indices_.push_back(binding_index);
        output_buffers_.push_back(output);
    }

    if (input_index_ < 0 || output_buffers_.empty()) {
        throw std::runtime_error("TensorRT Engine 的输入输出数量不符合 DeepSORT Orin 示例要求");
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
