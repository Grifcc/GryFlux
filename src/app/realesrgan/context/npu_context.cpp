#include "context/npu_context.h"

#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace
{
constexpr int kRknnTensorTypeFp16 = 1; // RKNN_TENSOR_FLOAT16 in current rknn_api enum layout

int checkRknnCall(int ret, const char *op)
{
    if (ret < 0)
    {
        throw std::runtime_error(std::string(op) + " failed, ret=" + std::to_string(ret));
    }
    return ret;
}

int getAttrHeight(const rknn_tensor_attr &attr)
{
    if (attr.n_dims >= 4)
    {
        return static_cast<int>(attr.dims[1]);
    }
    if (attr.n_dims == 3)
    {
        return static_cast<int>(attr.dims[0]);
    }
    return 0;
}

int getAttrWidth(const rknn_tensor_attr &attr)
{
    if (attr.n_dims >= 4)
    {
        return static_cast<int>(attr.dims[2]);
    }
    if (attr.n_dims == 3)
    {
        return static_cast<int>(attr.dims[1]);
    }
    return 0;
}

int getAttrChannels(const rknn_tensor_attr &attr)
{
    if (attr.n_dims >= 4)
    {
        return static_cast<int>(attr.dims[3]);
    }
    if (attr.n_dims == 3)
    {
        return static_cast<int>(attr.dims[2]);
    }
    return 0;
}

bool isFloatTensorType(rknn_tensor_type type)
{
    if (type == RKNN_TENSOR_FLOAT32)
    {
        return true;
    }
    if (static_cast<int>(type) == kRknnTensorTypeFp16)
    {
        return true;
    }
    return false;
}
} // namespace

NpuContext::NpuContext(int deviceId,
                       const std::string &modelPath,
                       int expectedModelWidth,
                       int expectedModelHeight)
    : deviceId_(deviceId),
      modelPath_(modelPath),
      expectedModelWidth_(expectedModelWidth),
      expectedModelHeight_(expectedModelHeight)
{
    std::ifstream ifs(modelPath_, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Failed to open RKNN model: " + modelPath_);
    }

    ifs.seekg(0, std::ios::end);
    const auto len = ifs.tellg();
    if (len <= 0)
    {
        throw std::runtime_error("Empty RKNN model: " + modelPath_);
    }

    ifs.seekg(0, std::ios::beg);
    modelData_.resize(static_cast<size_t>(len));
    ifs.read(reinterpret_cast<char *>(modelData_.data()), static_cast<std::streamsize>(len));

    checkRknnCall(rknn_init(&ctx_, modelData_.data(), modelData_.size(), 0, nullptr), "rknn_init");
    checkRknnCall(rknn_set_core_mask(ctx_, toCoreMask(deviceId_)), "rknn_set_core_mask");

    rknn_input_output_num ioNum{};
    checkRknnCall(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query(io_num)");

    inputNum_ = static_cast<int>(ioNum.n_input);
    outputNum_ = static_cast<int>(ioNum.n_output);

    if (inputNum_ != 1)
    {
        throw std::runtime_error("Only single-input model is supported, got n_input=" + std::to_string(inputNum_));
    }
    if (outputNum_ < 1)
    {
        throw std::runtime_error("Model has no output tensor");
    }

    inputAttrs_.resize(inputNum_);
    for (int i = 0; i < inputNum_; ++i)
    {
        std::memset(&inputAttrs_[i], 0, sizeof(rknn_tensor_attr));
        inputAttrs_[i].index = static_cast<uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &inputAttrs_[i], sizeof(rknn_tensor_attr)),
                      "rknn_query(input_attr)");
    }

    auto &inputAttr = inputAttrs_[0];
    if (inputAttr.fmt != RKNN_TENSOR_NHWC)
    {
        throw std::runtime_error("Only NHWC input tensor is supported");
    }

    inputWidth_ = getAttrWidth(inputAttr);
    inputHeight_ = getAttrHeight(inputAttr);
    inputChannels_ = getAttrChannels(inputAttr);

    if (inputWidth_ <= 0 || inputHeight_ <= 0 || inputChannels_ <= 0)
    {
        throw std::runtime_error("Invalid input tensor shape from RKNN model");
    }

    if (inputWidth_ != expectedModelWidth_ || inputHeight_ != expectedModelHeight_)
    {
        throw std::runtime_error("Model input shape mismatch. expected=" +
                                 std::to_string(expectedModelWidth_) + "x" +
                                 std::to_string(expectedModelHeight_) +
                                 ", model=" + std::to_string(inputWidth_) + "x" + std::to_string(inputHeight_));
    }

    inputType_ = inputAttr.type;
    inputQuantized_ = ((inputType_ == RKNN_TENSOR_UINT8 || inputType_ == RKNN_TENSOR_INT8) &&
                       inputAttr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);
    inputScale_ = inputAttr.scale;
    inputZeroPoint_ = inputAttr.zp;

    LOG.info("RKNN input attr: type=%d qnt_type=%d fmt=%d dims=[%d,%d,%d,%d] scale=%f zp=%d",
             static_cast<int>(inputType_),
             static_cast<int>(inputAttr.qnt_type),
             static_cast<int>(inputAttr.fmt),
             static_cast<int>(inputAttr.dims[0]),
             static_cast<int>(inputAttr.dims[1]),
             static_cast<int>(inputAttr.dims[2]),
             static_cast<int>(inputAttr.dims[3]),
             static_cast<double>(inputScale_),
             inputZeroPoint_);

    if (inputType_ == RKNN_TENSOR_UINT8)
    {
        inputElementSize_ = sizeof(uint8_t);
    }
    else if (inputType_ == RKNN_TENSOR_INT8)
    {
        inputElementSize_ = sizeof(int8_t);
    }
    else if (inputType_ == RKNN_TENSOR_FLOAT32)
    {
        inputElementSize_ = sizeof(float);
    }
    else if (static_cast<int>(inputType_) == kRknnTensorTypeFp16)
    {
        inputElementSize_ = sizeof(uint16_t);
    }
    else
    {
        throw std::runtime_error("Unsupported input tensor type: " + std::to_string(static_cast<int>(inputType_)));
    }

    inputMems_.resize(inputNum_, nullptr);
    inputAttrs_[0].fmt = RKNN_TENSOR_NHWC;
    inputAttrs_[0].type = inputType_;

    inputMems_[0] = rknn_create_mem(ctx_, inputAttrs_[0].size_with_stride);
    if (!inputMems_[0])
    {
        throw std::runtime_error("rknn_create_mem(input) failed");
    }
    checkRknnCall(rknn_set_io_mem(ctx_, inputMems_[0], &inputAttrs_[0]), "rknn_set_io_mem(input)");

    outputAttrs_.resize(outputNum_);
    outputMems_.resize(outputNum_, nullptr);
    outputQuantized_.resize(outputNum_, false);
    outputCache_.resize(outputNum_);

    for (int i = 0; i < outputNum_; ++i)
    {
        std::memset(&outputAttrs_[i], 0, sizeof(rknn_tensor_attr));
        outputAttrs_[i].index = static_cast<uint32_t>(i);
        checkRknnCall(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &outputAttrs_[i], sizeof(rknn_tensor_attr)),
                      "rknn_query(output_attr)");

        const bool quantized = (outputAttrs_[i].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                                !isFloatTensorType(outputAttrs_[i].type));
        outputQuantized_[i] = quantized;

        outputAttrs_[i].type = quantized ? RKNN_TENSOR_INT8 : RKNN_TENSOR_FLOAT32;
        const size_t outBytes = static_cast<size_t>(outputAttrs_[i].n_elems) *
                                (quantized ? sizeof(int8_t) : sizeof(float));

        outputMems_[i] = rknn_create_mem(ctx_, outBytes);
        if (!outputMems_[i])
        {
            throw std::runtime_error("rknn_create_mem(output) failed at index " + std::to_string(i));
        }

        checkRknnCall(rknn_set_io_mem(ctx_, outputMems_[i], &outputAttrs_[i]), "rknn_set_io_mem(output)");
        outputCache_[i].reset(new float[outputAttrs_[i].n_elems], std::default_delete<float[]>());

        LOG.info("RKNN output[%d] attr: type=%d qnt_type=%d fmt=%d dims=[%d,%d,%d,%d] n_elems=%d scale=%f zp=%d",
                 i,
                 static_cast<int>(outputAttrs_[i].type),
                 static_cast<int>(outputAttrs_[i].qnt_type),
                 static_cast<int>(outputAttrs_[i].fmt),
                 static_cast<int>(outputAttrs_[i].dims[0]),
                 static_cast<int>(outputAttrs_[i].dims[1]),
                 static_cast<int>(outputAttrs_[i].dims[2]),
                 static_cast<int>(outputAttrs_[i].dims[3]),
                 static_cast<int>(outputAttrs_[i].n_elems),
                 static_cast<double>(outputAttrs_[i].scale),
                 outputAttrs_[i].zp);
    }

    LOG.info("NpuContext initialized: device=%d model=%s input=%dx%dx%d outputs=%d",
             deviceId_,
             modelPath_.c_str(),
             inputWidth_,
             inputHeight_,
             inputChannels_,
             outputNum_);
}

NpuContext::~NpuContext()
{
    for (auto *m : inputMems_)
    {
        if (m)
        {
            rknn_destroy_mem(ctx_, m);
        }
    }

    for (auto *m : outputMems_)
    {
        if (m)
        {
            rknn_destroy_mem(ctx_, m);
        }
    }

    if (ctx_)
    {
        rknn_destroy(ctx_);
    }
}

void NpuContext::run(const cv::Mat &inputRgbU8, cv::Mat &srTensorF32)
{
    if (inputRgbU8.empty())
    {
        throw std::runtime_error("NpuContext::run got empty input");
    }
    if (inputRgbU8.type() != CV_8UC3)
    {
        throw std::runtime_error("NpuContext::run expects CV_8UC3 RGB input");
    }
    if (inputRgbU8.cols != expectedModelWidth_ || inputRgbU8.rows != expectedModelHeight_)
    {
        throw std::runtime_error("NpuContext::run input shape mismatch");
    }
    if (inputRgbU8.channels() != inputChannels_)
    {
        throw std::runtime_error("NpuContext::run input channel mismatch");
    }

    cv::Mat prepared;

    if (inputQuantized_)
    {
        cv::Mat normalized;
        inputRgbU8.convertTo(normalized, CV_32FC3, 1.0 / 255.0);
        cv::max(normalized, 0.0, normalized);
        cv::min(normalized, 1.0, normalized);

        const int matType = (inputType_ == RKNN_TENSOR_UINT8)
                                ? CV_MAKETYPE(CV_8U, inputChannels_)
                                : CV_MAKETYPE(CV_8S, inputChannels_);
        prepared.create(normalized.rows, normalized.cols, matType);

        const float invScale = (inputScale_ == 0.0f) ? 0.0f : (1.0f / inputScale_);
        const int minVal = (inputType_ == RKNN_TENSOR_UINT8) ? 0 : -128;
        const int maxVal = (inputType_ == RKNN_TENSOR_UINT8) ? 255 : 127;

        for (int r = 0; r < normalized.rows; ++r)
        {
            const float *srcRow = normalized.ptr<float>(r);

            if (inputType_ == RKNN_TENSOR_UINT8)
            {
                auto *dstRow = prepared.ptr<uint8_t>(r);
                for (int c = 0; c < normalized.cols; ++c)
                {
                    for (int ch = 0; ch < inputChannels_; ++ch)
                    {
                        const float realVal = srcRow[c * inputChannels_ + ch];
                        const int q = static_cast<int>(std::round(realVal * invScale + static_cast<float>(inputZeroPoint_)));
                        const int clamped = std::max(minVal, std::min(maxVal, q));
                        dstRow[c * inputChannels_ + ch] = static_cast<uint8_t>(clamped);
                    }
                }
            }
            else
            {
                auto *dstRow = prepared.ptr<int8_t>(r);
                for (int c = 0; c < normalized.cols; ++c)
                {
                    for (int ch = 0; ch < inputChannels_; ++ch)
                    {
                        const float realVal = srcRow[c * inputChannels_ + ch];
                        const int q = static_cast<int>(std::round(realVal * invScale + static_cast<float>(inputZeroPoint_)));
                        const int clamped = std::max(minVal, std::min(maxVal, q));
                        dstRow[c * inputChannels_ + ch] = static_cast<int8_t>(clamped);
                    }
                }
            }
        }
    }
    else
    {
        if (inputType_ == RKNN_TENSOR_FLOAT32)
        {
            cv::Mat fp32;
            inputRgbU8.convertTo(fp32, CV_32FC3, 1.0 / 255.0);
            cv::max(fp32, 0.0, fp32);
            cv::min(fp32, 1.0, fp32);
            prepared = fp32;
        }
        
        else if (static_cast<int>(inputType_) == kRknnTensorTypeFp16)
        {
#if defined(CV_16F)
            cv::Mat fp32;
            // For FP16 non-quantized model, keep 0..255 dynamic range (legacy RealESRGAN behavior).
            inputRgbU8.convertTo(fp32, CV_32FC3, 1.0);
            cv::max(fp32, 0.0, fp32);
            cv::min(fp32, 255.0, fp32);
            fp32.convertTo(prepared, CV_MAKETYPE(CV_16F, inputChannels_));
#else
            throw std::runtime_error("OpenCV without CV_16F support cannot feed FLOAT16 RKNN input");
#endif
        }
        else if (inputType_ == RKNN_TENSOR_UINT8)
        {
            prepared = inputRgbU8;
        }
        else
        {
            throw std::runtime_error("Unsupported non-quantized input tensor type");
        }
    }

    if (!prepared.isContinuous())
    {
        prepared = prepared.clone();
    }

    const size_t srcBytes = prepared.total() * prepared.elemSize();
    const size_t dstBytes = static_cast<size_t>(inputMems_[0]->size);

    std::memset(inputMems_[0]->virt_addr, 0, dstBytes);
    std::memcpy(inputMems_[0]->virt_addr, prepared.data, std::min(srcBytes, dstBytes));

#if defined(RKNN_VERSION_DEF) && (RKNN_VERSION_DEF >= 160)
    checkRknnCall(rknn_mem_sync(ctx_, inputMems_[0], RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync(input)");
#endif

    checkRknnCall(rknn_run(ctx_, nullptr), "rknn_run");

    for (int i = 0; i < outputNum_; ++i)
    {
#if defined(RKNN_VERSION_DEF) && (RKNN_VERSION_DEF >= 160)
        checkRknnCall(rknn_mem_sync(ctx_, outputMems_[i], RKNN_MEMORY_SYNC_FROM_DEVICE), "rknn_mem_sync(output)");
#endif

        float *dst = outputCache_[i].get();
        const int elemCount = static_cast<int>(outputAttrs_[i].n_elems);

        if (outputQuantized_[i])
        {
            const int8_t *raw = reinterpret_cast<const int8_t *>(outputMems_[i]->virt_addr);
            for (int j = 0; j < elemCount; ++j)
            {
                dst[j] = dequantize(raw[j], outputAttrs_[i].zp, outputAttrs_[i].scale);
            }
        }
        else
        {
            const float *raw = reinterpret_cast<const float *>(outputMems_[i]->virt_addr);
            std::memcpy(dst, raw, static_cast<size_t>(elemCount) * sizeof(float));
        }
    }

    const auto &outAttr = outputAttrs_[0];
    const float *outData = outputCache_[0].get();

    int outC = 0;
    int outH = 0;
    int outW = 0;

    if (outAttr.n_dims >= 4)
    {
        if (outAttr.fmt == RKNN_TENSOR_NHWC)
        {
            outH = static_cast<int>(outAttr.dims[1]);
            outW = static_cast<int>(outAttr.dims[2]);
            outC = static_cast<int>(outAttr.dims[3]);
        }
        else
        {
            outC = static_cast<int>(outAttr.dims[1]);
            outH = static_cast<int>(outAttr.dims[2]);
            outW = static_cast<int>(outAttr.dims[3]);
        }
    }
    else if (outAttr.n_dims == 3)
    {
        if (outAttr.fmt == RKNN_TENSOR_NHWC)
        {
            outH = static_cast<int>(outAttr.dims[0]);
            outW = static_cast<int>(outAttr.dims[1]);
            outC = static_cast<int>(outAttr.dims[2]);
        }
        else
        {
            outC = static_cast<int>(outAttr.dims[0]);
            outH = static_cast<int>(outAttr.dims[1]);
            outW = static_cast<int>(outAttr.dims[2]);
        }
    }

    if (outC <= 0 || outH <= 0 || outW <= 0)
    {
        throw std::runtime_error("Invalid output tensor shape from RKNN model");
    }

    if (outAttr.fmt == RKNN_TENSOR_NHWC)
    {
        srTensorF32 = makeOutputMatFromNHWC(outData, outH, outW, outC);
    }
    else
    {
        srTensorF32 = makeOutputMatFromNCHW(outData, outC, outH, outW);
    }
}

float NpuContext::dequantize(int8_t qnt, int zp, float scale)
{
    return (static_cast<float>(qnt) - static_cast<float>(zp)) * scale;
}

rknn_core_mask NpuContext::toCoreMask(int deviceId)
{
    static const rknn_core_mask kMasks[] = {
        RKNN_NPU_CORE_0,
        RKNN_NPU_CORE_1,
        RKNN_NPU_CORE_2,
        RKNN_NPU_CORE_0_1,
        RKNN_NPU_CORE_0_1_2,
        RKNN_NPU_CORE_AUTO,
    };

    if (deviceId < 0)
    {
        deviceId = 0;
    }

    if (deviceId >= static_cast<int>(sizeof(kMasks) / sizeof(kMasks[0])))
    {
        deviceId = static_cast<int>(sizeof(kMasks) / sizeof(kMasks[0])) - 1;
    }

    return kMasks[deviceId];
}

cv::Mat NpuContext::makeOutputMatFromNCHW(const float *data, int c, int h, int w) const
{
    std::vector<cv::Mat> channels;
    channels.reserve(static_cast<size_t>(c));

    const size_t planeSize = static_cast<size_t>(h) * static_cast<size_t>(w);
    for (int i = 0; i < c; ++i)
    {
        const float *ptr = data + static_cast<size_t>(i) * planeSize;
        cv::Mat ch(h, w, CV_32F, const_cast<float *>(ptr));
        channels.emplace_back(ch.clone());
    }

    cv::Mat merged;
    cv::merge(channels, merged);
    return merged;
}

cv::Mat NpuContext::makeOutputMatFromNHWC(const float *data, int h, int w, int c) const
{
    const int type = CV_MAKETYPE(CV_32F, c);
    cv::Mat output(h, w, type, const_cast<float *>(data));
    return output.clone();
}
