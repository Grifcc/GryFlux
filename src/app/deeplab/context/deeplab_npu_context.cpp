#include "context/deeplab_npu_context.h"

#include "utils/logger.h"

#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
constexpr rknn_core_mask kCoreMasks[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0_1,
    RKNN_NPU_CORE_0_1_2,
};

int checkRknnCall(int ret, const char *op)
{
    if (ret < 0)
    {
        throw std::runtime_error(std::string(op) + " failed, ret=" + std::to_string(ret));
    }
    return ret;
}
} // namespace

DeeplabNpuContext::DeeplabNpuContext(int deviceId,
                                     const std::string &modelPath,
                                     int expectedModelWidth,
                                     int expectedModelHeight)
    : deviceId_(deviceId),
      modelPath_(modelPath),
      expectedModelWidth_(expectedModelWidth),
      expectedModelHeight_(expectedModelHeight)
{
    std::ifstream modelStream(modelPath_, std::ios::binary);
    if (!modelStream)
    {
        throw std::runtime_error("Failed to open RKNN model: " + modelPath_);
    }

    modelStream.seekg(0, std::ios::end);
    const auto length = modelStream.tellg();
    if (length <= 0)
    {
        throw std::runtime_error("Empty RKNN model: " + modelPath_);
    }

    modelStream.seekg(0, std::ios::beg);
    modelData_.resize(static_cast<std::size_t>(length));
    modelStream.read(reinterpret_cast<char *>(modelData_.data()), static_cast<std::streamsize>(length));
    if (!modelStream)
    {
        throw std::runtime_error("Failed to read RKNN model: " + modelPath_);
    }

    checkRknnCall(rknn_init(&ctx_, modelData_.data(), modelData_.size(), 0, nullptr), "rknn_init");
    checkRknnCall(rknn_set_core_mask(ctx_, toCoreMask(deviceId_)), "rknn_set_core_mask");

    rknn_input_output_num ioNum{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query(io_num)");

    if (ioNum.n_input != 1)
    {
        throw std::runtime_error("Deeplab only supports single-input RKNN models");
    }
    if (ioNum.n_output < 1)
    {
        throw std::runtime_error("Deeplab model has no outputs");
    }

    std::memset(&inputAttr_, 0, sizeof(inputAttr_));
    inputAttr_.index = 0;
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputAttr_, sizeof(inputAttr_)), "rknn_query(input_attr)");

    resolveInputShape(inputAttr_, inputHeight_, inputWidth_, inputChannels_);
    if (inputAttr_.fmt != RKNN_TENSOR_NHWC)
    {
        throw std::runtime_error("Deeplab expects RKNN input tensor format NHWC");
    }
    if (inputWidth_ != expectedModelWidth_ || inputHeight_ != expectedModelHeight_ || inputChannels_ != 3)
    {
        throw std::runtime_error("RKNN input shape mismatch for Deeplab");
    }

    inputType_ = inputAttr_.type;
    inputScale_ = (inputAttr_.scale == 0.0f) ? 1.0f : inputAttr_.scale;
    inputZeroPoint_ = inputAttr_.zp;

    LOG.info("Deeplab RKNN input: fmt=%d dims=[%d,%d,%d,%d] n_dims=%u n_elems=%u type=%d qnt_type=%d zp=%d scale=%f size=%u stride=%u",
             static_cast<int>(inputAttr_.fmt),
             static_cast<int>(inputAttr_.dims[0]),
             static_cast<int>(inputAttr_.dims[1]),
             static_cast<int>(inputAttr_.dims[2]),
             static_cast<int>(inputAttr_.dims[3]),
             static_cast<unsigned int>(inputAttr_.n_dims),
             static_cast<unsigned int>(inputAttr_.n_elems),
             static_cast<int>(inputType_),
             static_cast<int>(inputAttr_.qnt_type),
             inputZeroPoint_,
             static_cast<double>(inputScale_),
             static_cast<unsigned int>(inputAttr_.size),
             static_cast<unsigned int>(inputAttr_.size_with_stride));

    outputAttrs_.resize(ioNum.n_output);

    for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
    {
        auto &attr = outputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<std::uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query(output_attr)");

        LOG.info("Deeplab RKNN output[%zu]: fmt=%d dims=[%d,%d,%d,%d] n_dims=%u n_elems=%u type=%d qnt_type=%d zp=%d scale=%f",
                 i,
                 static_cast<int>(attr.fmt),
                 static_cast<int>(attr.dims[0]),
                 static_cast<int>(attr.dims[1]),
                 static_cast<int>(attr.dims[2]),
                 static_cast<int>(attr.dims[3]),
                 static_cast<unsigned int>(attr.n_dims),
                 static_cast<unsigned int>(attr.n_elems),
                 static_cast<int>(attr.type),
                 static_cast<int>(attr.qnt_type),
                 attr.zp,
                 static_cast<double>(attr.scale));
    }

    initialized_ = true;
}

DeeplabNpuContext::~DeeplabNpuContext()
{
    releaseResources();
}

void DeeplabNpuContext::run(const std::vector<uint8_t> &inputData,
                            std::vector<DeeplabPacket::InferenceOutput> &outputs)
{
    if (!initialized_)
    {
        throw std::runtime_error("DeeplabNpuContext is not initialized");
    }
    const std::size_t expectedBytes = static_cast<std::size_t>(inputWidth_) * static_cast<std::size_t>(inputHeight_) *
                                      static_cast<std::size_t>(inputChannels_);
    if (inputData.size() != expectedBytes)
    {
        throw std::runtime_error("Invalid Deeplab input tensor byte size");
    }

    rknn_input input{};
    input.index = 0;
    input.buf = const_cast<uint8_t *>(inputData.data());
    input.size = static_cast<uint32_t>(inputData.size());
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    checkRknnCall(rknn_inputs_set(ctx_, 1, &input), "rknn_inputs_set");
    checkRknnCall(rknn_run(ctx_, nullptr), "rknn_run");

    std::vector<rknn_output> rknnOutputs(outputAttrs_.size());
    for (std::size_t i = 0; i < rknnOutputs.size(); ++i)
    {
        rknnOutputs[i].want_float = 1;
        rknnOutputs[i].is_prealloc = 0;
        rknnOutputs[i].index = static_cast<uint32_t>(i);
    }
    checkRknnCall(rknn_outputs_get(ctx_, static_cast<uint32_t>(rknnOutputs.size()), rknnOutputs.data(), nullptr),
                  "rknn_outputs_get");

    outputs.resize(outputAttrs_.size());
    for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
    {
        const auto &attr = outputAttrs_[i];
        auto &output = outputs[i];
        int gridH = 0;
        int gridW = 0;
        if (attr.n_dims >= 4)
        {
            gridH = static_cast<int>(attr.dims[1]);
            gridW = static_cast<int>(attr.dims[2]);
        }
        else if (attr.n_dims == 3)
        {
            gridH = static_cast<int>(attr.dims[0]);
            gridW = static_cast<int>(attr.dims[1]);
        }
        if (gridH <= 0 || gridW <= 0)
        {
            checkRknnCall(rknn_outputs_release(ctx_, static_cast<uint32_t>(rknnOutputs.size()), rknnOutputs.data()),
                          "rknn_outputs_release");
            throw std::runtime_error("Invalid Deeplab output tensor shape");
        }

        const std::size_t floatCount = static_cast<std::size_t>(rknnOutputs[i].size) / sizeof(float);
        const std::size_t channels = floatCount / (static_cast<std::size_t>(gridH) * static_cast<std::size_t>(gridW));
        if (channels == 0 || channels * static_cast<std::size_t>(gridH) * static_cast<std::size_t>(gridW) != floatCount)
        {
            checkRknnCall(rknn_outputs_release(ctx_, static_cast<uint32_t>(rknnOutputs.size()), rknnOutputs.data()),
                          "rknn_outputs_release");
            throw std::runtime_error("Unexpected Deeplab output float buffer size");
        }

        output.gridH = static_cast<std::size_t>(gridH);
        output.gridW = static_cast<std::size_t>(gridW);
        output.channels = channels;
        output.data.resize(floatCount);
        const auto *src = static_cast<const float *>(rknnOutputs[i].buf);
        std::copy(src, src + floatCount, output.data.begin());
    }

    checkRknnCall(rknn_outputs_release(ctx_, static_cast<uint32_t>(rknnOutputs.size()), rknnOutputs.data()),
                  "rknn_outputs_release");
}

rknn_core_mask DeeplabNpuContext::toCoreMask(int deviceId)
{
    if (deviceId < 0)
    {
        deviceId = 0;
    }
    if (deviceId >= static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])))
    {
        deviceId = static_cast<int>(sizeof(kCoreMasks) / sizeof(kCoreMasks[0])) - 1;
    }
    return kCoreMasks[deviceId];
}

void DeeplabNpuContext::resolveInputShape(const rknn_tensor_attr &attr, int &height, int &width, int &channels)
{
    height = 0;
    width = 0;
    channels = 0;
    if (attr.n_dims >= 4)
    {
        height = static_cast<int>(attr.dims[1]);
        width = static_cast<int>(attr.dims[2]);
        channels = static_cast<int>(attr.dims[3]);
    }
}

void DeeplabNpuContext::releaseResources()
{
    if (initialized_)
    {
        rknn_destroy(ctx_);
        initialized_ = false;
    }
}
