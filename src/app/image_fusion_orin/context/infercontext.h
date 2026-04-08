#pragma once

#include "framework/context.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class InferContext : public GryFlux::Context {
public:
    struct TensorBinding {
        int bindingIndex = -1;
        std::string name;
        nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT;
        size_t byteSize = 0;
        size_t elementCount = 0;
        void* devicePtr = nullptr;
        std::vector<std::uint8_t> hostBuffer;
    };

    InferContext();
    ~InferContext() override;

    bool Init(const std::string& modelPath, int deviceId);
    void Destroy();
    void bindCurrentThread();

    size_t GetInputElementCount(size_t index) const { return inputBindings_.at(index).elementCount; }
    size_t GetOutputElementCount() const { return outputBinding_.elementCount; }
    void copyInputToDevice(size_t index, const float* hostData, size_t elementCount);
    void execute();
    void copyOutputToHost(float* hostData, size_t elementCount);

private:
    void loadEngine(const std::string& modelPath);
    bool allocateBuffers();
    void logBindings() const;

    template <typename T>
    struct TrtDeleter {
        void operator()(T* ptr) const;
    };

    int32_t deviceId_;
    cudaStream_t stream_ = nullptr;

    std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;

    std::vector<void*> bindings_;
    std::vector<TensorBinding> inputBindings_;
    TensorBinding outputBinding_;
};
