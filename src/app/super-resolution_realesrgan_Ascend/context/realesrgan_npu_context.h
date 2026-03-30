#pragma once

#include "framework/context.h"

#include "acl/acl.h"

#include <array>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#define CHECK_ACL(ret, msg) \
    if ((ret) != ACL_SUCCESS) { \
        throw std::runtime_error(std::string("ACL Error ") + std::to_string(static_cast<int>(ret)) + ": " + (msg)); \
    }

class RealEsrganNPUContext : public GryFlux::Context
{
public:
    RealEsrganNPUContext(int deviceId, const std::string &modelPath)
        : deviceId_(deviceId)
    {
        std::cout << "[RealEsrganNPUContext] Init device " << deviceId_ << "..." << std::endl;

        CHECK_ACL(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        CHECK_ACL(aclrtCreateContext(&context_, deviceId_), "aclrtCreateContext");
        CHECK_ACL(aclrtCreateStream(&stream_), "aclrtCreateStream");

        CHECK_ACL(aclmdlLoadFromFile(modelPath.c_str(), &modelId_), "aclmdlLoadFromFile");
        modelDesc_ = aclmdlCreateDesc();
        CHECK_ACL(aclmdlGetDesc(modelDesc_, modelId_), "aclmdlGetDesc");

        CHECK_ACL(aclmdlGetInputDims(modelDesc_, 0, &inputDims_), "aclmdlGetInputDims");
        CHECK_ACL(aclmdlGetOutputDims(modelDesc_, 0, &outputDims_), "aclmdlGetOutputDims");
        currentOutputDims_ = outputDims_;

        inputFormat_ = aclmdlGetInputFormat(modelDesc_, 0);
        outputFormat_ = aclmdlGetOutputFormat(modelDesc_, 0);
        inputDataType_ = aclmdlGetInputDataType(modelDesc_, 0);
        outputDataType_ = aclmdlGetOutputDataType(modelDesc_, 0);

        inputSize_ = aclmdlGetInputSizeByIndex(modelDesc_, 0);
        outputSize_ = aclmdlGetOutputSizeByIndex(modelDesc_, 0);

        CHECK_ACL(aclrtMalloc(&devBufIn_, inputSize_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc dev_in");
        CHECK_ACL(aclrtMalloc(&devBufOut_, outputSize_, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc dev_out");

        inputDataset_ = aclmdlCreateDataset();
        aclDataBuffer *inBuf = aclCreateDataBuffer(devBufIn_, inputSize_);
        CHECK_ACL(aclmdlAddDatasetBuffer(inputDataset_, inBuf), "Add input buf");

        outputDataset_ = aclmdlCreateDataset();
        aclDataBuffer *outBuf = aclCreateDataBuffer(devBufOut_, outputSize_);
        CHECK_ACL(aclmdlAddDatasetBuffer(outputDataset_, outBuf), "Add output buf");

        std::cout << "[RealEsrganNPUContext] Input dims=" << dimsToString(inputDims_)
                  << ", format=" << formatToString(inputFormat_)
                  << ", dtype=" << dataTypeToString(inputDataType_)
                  << ", bytes=" << inputSize_ << std::endl;
        std::cout << "[RealEsrganNPUContext] Output dims=" << dimsToString(outputDims_)
                  << ", format=" << formatToString(outputFormat_)
                  << ", dtype=" << dataTypeToString(outputDataType_)
                  << ", bytes=" << outputSize_ << std::endl;
    }

    ~RealEsrganNPUContext() override
    {
        std::cout << "[RealEsrganNPUContext] Release device " << deviceId_ << "..." << std::endl;
        aclrtSetCurrentContext(context_);

        if (inputDataset_) aclmdlDestroyDataset(inputDataset_);
        if (outputDataset_) aclmdlDestroyDataset(outputDataset_);
        if (devBufIn_) aclrtFree(devBufIn_);
        if (devBufOut_) aclrtFree(devBufOut_);
        if (modelDesc_) aclmdlDestroyDesc(modelDesc_);
        if (modelId_) aclmdlUnload(modelId_);
        if (stream_) aclrtDestroyStream(stream_);
        if (context_) aclrtDestroyContext(context_);

        aclrtResetDevice(deviceId_);
    }

    void bindToCurrentThread()
    {
        CHECK_ACL(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
    }

    void refreshCurrentOutputDims()
    {
        aclmdlIODims runtimeDims{};
        CHECK_ACL(aclmdlGetCurOutputDims(modelDesc_, 0, &runtimeDims), "aclmdlGetCurOutputDims");
        currentOutputDims_ = runtimeDims;

        if (!hasLoggedCurrentOutputDims_ || !sameDims(currentOutputDims_, lastLoggedOutputDims_))
        {
            std::cout << "[RealEsrganNPUContext] Runtime output dims=" << dimsToString(currentOutputDims_)
                      << std::endl;
            lastLoggedOutputDims_ = currentOutputDims_;
            hasLoggedCurrentOutputDims_ = true;
        }
    }

    int getDeviceId() const { return deviceId_; }
    aclrtStream getStream() const { return stream_; }
    uint32_t getModelId() const { return modelId_; }
    aclmdlDataset *getInputDataset() const { return inputDataset_; }
    aclmdlDataset *getOutputDataset() const { return outputDataset_; }
    void *getDevBufIn() const { return devBufIn_; }
    void *getDevBufOut() const { return devBufOut_; }
    size_t getInputSize() const { return inputSize_; }
    size_t getOutputSize() const { return outputSize_; }
    const aclmdlIODims &getInputDims() const { return inputDims_; }
    const aclmdlIODims &getOutputDims() const { return outputDims_; }
    const aclmdlIODims &getCurrentOutputDims() const { return currentOutputDims_; }
    aclFormat getInputFormat() const { return inputFormat_; }
    aclFormat getOutputFormat() const { return outputFormat_; }
    aclDataType getInputDataType() const { return inputDataType_; }
    aclDataType getOutputDataType() const { return outputDataType_; }

private:
    static bool sameDims(const aclmdlIODims &lhs, const aclmdlIODims &rhs)
    {
        if (lhs.dimCount != rhs.dimCount)
        {
            return false;
        }
        for (size_t i = 0; i < lhs.dimCount; ++i)
        {
            if (lhs.dims[i] != rhs.dims[i])
            {
                return false;
            }
        }
        return true;
    }

    static std::string dimsToString(const aclmdlIODims &dims)
    {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < dims.dimCount; ++i)
        {
            if (i != 0)
            {
                oss << ", ";
            }
            oss << dims.dims[i];
        }
        oss << "]";
        return oss.str();
    }

    static const char *formatToString(aclFormat format)
    {
        switch (format)
        {
            case ACL_FORMAT_NCHW:
                return "NCHW";
            case ACL_FORMAT_NHWC:
                return "NHWC";
            case ACL_FORMAT_ND:
                return "ND";
            case ACL_FORMAT_NC:
                return "NC";
            default:
                return "OTHER";
        }
    }

    static const char *dataTypeToString(aclDataType dataType)
    {
        switch (dataType)
        {
            case ACL_FLOAT:
                return "FLOAT32";
            case ACL_FLOAT16:
                return "FLOAT16";
            case ACL_UINT8:
                return "UINT8";
            case ACL_INT8:
                return "INT8";
            default:
                return "OTHER";
        }
    }

    int deviceId_ = 0;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    uint32_t modelId_ = 0;
    aclmdlDesc *modelDesc_ = nullptr;
    size_t inputSize_ = 0;
    size_t outputSize_ = 0;
    aclmdlIODims inputDims_{};
    aclmdlIODims outputDims_{};
    aclmdlIODims currentOutputDims_{};
    aclmdlIODims lastLoggedOutputDims_{};
    aclFormat inputFormat_ = ACL_FORMAT_UNDEFINED;
    aclFormat outputFormat_ = ACL_FORMAT_UNDEFINED;
    aclDataType inputDataType_ = ACL_DT_UNDEFINED;
    aclDataType outputDataType_ = ACL_DT_UNDEFINED;
    void *devBufIn_ = nullptr;
    void *devBufOut_ = nullptr;
    aclmdlDataset *inputDataset_ = nullptr;
    aclmdlDataset *outputDataset_ = nullptr;
    bool hasLoggedCurrentOutputDims_ = false;
};
