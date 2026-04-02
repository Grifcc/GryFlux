#include "context/fusion_npu_context.h"

#include "utils/logger.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>

namespace
{
constexpr rknn_core_mask kCoreMasks[] = {
    RKNN_NPU_CORE_0,
    RKNN_NPU_CORE_1,
    RKNN_NPU_CORE_2,
    RKNN_NPU_CORE_0_1,
    RKNN_NPU_CORE_0_1_2,
};

#define RKNN_CHECK(op, msg)                                                                                              \
    do                                                                                                                   \
    {                                                                                                                    \
        const int ret = (op);                                                                                            \
        if (ret < 0)                                                                                                     \
        {                                                                                                                \
            LOG.error("[FusionNpuContext] %s failed with ret=%d", msg, ret);                                             \
            throw std::runtime_error(msg);                                                                               \
        }                                                                                                                \
    } while (0)
} // namespace

FusionNpuContext::FusionNpuContext(int deviceId,
                                   const std::string &modelPath,
                                   int expectedModelWidth,
                                   int expectedModelHeight)
    : deviceId_(deviceId),
      modelPath_(modelPath),
      expectedModelWidth_(expectedModelWidth),
      expectedModelHeight_(expectedModelHeight)
{
    auto modelData = loadModel(modelPath_);
    int initRet = rknn_init(&ctx_, modelData.first.get(), modelData.second, 0, nullptr);
    if (initRet < 0)
    {
        LOG.error("[FusionNpuContext] rknn_init failed with ret=%d for model=%s", initRet, modelPath_.c_str());
        throw std::runtime_error("rknn_init");
    }

    RKNN_CHECK(rknn_set_core_mask(ctx_, toCoreMask(deviceId_)), "rknn_set_core_mask");
    prepareTensorAttributes();
    initialized_ = true;
}

FusionNpuContext::~FusionNpuContext()
{
    releaseResources();
}

void FusionNpuContext::run(const cv::Mat &visYF32,
                           const cv::Mat &infraredF32,
                           cv::Mat &fusedYF32)
{
    if (!initialized_)
    {
        throw std::runtime_error("FusionNpuContext is not initialized");
    }
    if (visYF32.empty())
    {
        throw std::runtime_error("FusionNpuContext received empty visible Y input");
    }
    if (visYF32.type() != CV_32FC1)
    {
        throw std::runtime_error("FusionNpuContext expects visible Y as CV_32FC1");
    }
    if (visYF32.cols != expectedModelWidth_ || visYF32.rows != expectedModelHeight_)
    {
        throw std::runtime_error("Visible Y input size mismatch");
    }
    if (inputAttrs_.size() == 2)
    {
        if (infraredF32.empty())
        {
            throw std::runtime_error("FusionNpuContext requires infrared input for dual-input model");
        }
        if (infraredF32.type() != CV_32FC1)
        {
            throw std::runtime_error("FusionNpuContext expects infrared input as CV_32FC1");
        }
        if (infraredF32.cols != expectedModelWidth_ || infraredF32.rows != expectedModelHeight_)
        {
            throw std::runtime_error("Infrared input size mismatch");
        }
    }

    copyInputData(visYF32, 0);
    if (inputAttrs_.size() > 1)
    {
        copyInputData(infraredF32, 1);
    }

    RKNN_CHECK(rknn_run(ctx_, nullptr), "rknn_run");
    fusedYF32 = fetchOutputData(0);
}

rknn_core_mask FusionNpuContext::toCoreMask(int deviceId)
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

float FusionNpuContext::determineInputScaling(const rknn_tensor_attr &attr)
{
    const bool isFloat = (attr.type == RKNN_TENSOR_FLOAT16 || attr.type == RKNN_TENSOR_FLOAT32);
    if (!isFloat)
    {
        return 1.0f;
    }

    const bool affineLike = (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC ||
                             attr.qnt_type == RKNN_TENSOR_QNT_NONE);
    if (!affineLike)
    {
        return 1.0f;
    }

    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
        std::fabs(attr.scale - 1.0f) < 1e-4f &&
        attr.zp == 0)
    {
        return 255.0f;
    }

    return 1.0f;
}

std::size_t FusionNpuContext::tensorTypeSize(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return sizeof(float);
    case RKNN_TENSOR_FLOAT16:
        return sizeof(std::uint16_t);
    case RKNN_TENSOR_INT8:
        return sizeof(std::int8_t);
    case RKNN_TENSOR_UINT8:
        return sizeof(std::uint8_t);
    case RKNN_TENSOR_INT16:
        return sizeof(std::int16_t);
    case RKNN_TENSOR_UINT16:
        return sizeof(std::uint16_t);
    default:
        throw std::runtime_error("Unsupported tensor type");
    }
}

void FusionNpuContext::resolveSpatial(const rknn_tensor_attr &attr, int &height, int &width)
{
    if (attr.n_dims >= 4)
    {
        if (attr.fmt == RKNN_TENSOR_NCHW)
        {
            height = static_cast<int>(attr.dims[2]);
            width = static_cast<int>(attr.dims[3]);
        }
        else
        {
            height = static_cast<int>(attr.dims[1]);
            width = static_cast<int>(attr.dims[2]);
        }
        return;
    }

    if (attr.n_dims == 3)
    {
        if (attr.fmt == RKNN_TENSOR_NCHW)
        {
            height = static_cast<int>(attr.dims[1]);
            width = static_cast<int>(attr.dims[2]);
        }
        else
        {
            height = static_cast<int>(attr.dims[0]);
            width = static_cast<int>(attr.dims[1]);
        }
        return;
    }

    if (attr.n_dims == 2)
    {
        height = static_cast<int>(attr.dims[0]);
        width = static_cast<int>(attr.dims[1]);
    }
}

void FusionNpuContext::dumpTensorAttr(const rknn_tensor_attr &attr)
{
    LOG.info("[FusionNpuContext] index=%d name=%s dims=[%d,%d,%d,%d] size=%d type=%s qnt_type=%s zp=%d scale=%f",
             attr.index,
             attr.name,
             attr.dims[0],
             attr.dims[1],
             attr.dims[2],
             attr.dims[3],
             attr.size,
             get_type_string(attr.type),
             get_qnt_type_string(attr.qnt_type),
             attr.zp,
             attr.scale);
}

FusionNpuContext::ModelData FusionNpuContext::loadModel(const std::string &path) const
{
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin.is_open())
    {
        throw std::runtime_error("Failed to open RKNN model: " + path);
    }

    const auto fileSize = static_cast<std::size_t>(fin.tellg());
    fin.seekg(0, std::ios::beg);

    auto buffer = std::make_unique<unsigned char[]>(fileSize);
    fin.read(reinterpret_cast<char *>(buffer.get()), static_cast<std::streamsize>(fileSize));
    if (!fin)
    {
        throw std::runtime_error("Failed to read RKNN model: " + path);
    }

    return ModelData{std::move(buffer), fileSize};
}

void FusionNpuContext::prepareTensorAttributes()
{
    rknn_input_output_num ioNum{};
    RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum)), "rknn_query_in_out_num");

    if (ioNum.n_input != 1 && ioNum.n_input != 2)
    {
        throw std::runtime_error("FusionNetV2 only supports 1 or 2 model inputs");
    }
    if (ioNum.n_output < 1)
    {
        throw std::runtime_error("FusionNetV2 model has no outputs");
    }

    inputAttrs_.resize(ioNum.n_input);
    inputScaling_.clear();
    inputMems_.clear();
    inputMems_.reserve(ioNum.n_input);

    for (std::size_t i = 0; i < inputAttrs_.size(); ++i)
    {
        auto &attr = inputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<uint32_t>(i);
        RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr)), "rknn_query_input_attr");
        dumpTensorAttr(attr);

        int height = 0;
        int width = 0;
        resolveSpatial(attr, height, width);
        if (height != expectedModelHeight_ || width != expectedModelWidth_)
        {
            throw std::runtime_error("FusionNetV2 model input shape mismatch");
        }

        inputScaling_.push_back(determineInputScaling(attr));

        auto *tensorMem = rknn_create_mem(ctx_, attr.size);
        if (!tensorMem)
        {
            throw std::runtime_error("Failed to allocate RKNN input tensor memory");
        }

        inputMems_.push_back(tensorMem);
        RKNN_CHECK(rknn_set_io_mem(ctx_, tensorMem, &attr), "rknn_set_input_mem");
    }

    outputAttrs_.resize(ioNum.n_output);
    outputMems_.clear();
    outputMems_.reserve(ioNum.n_output);

    for (std::size_t i = 0; i < outputAttrs_.size(); ++i)
    {
        auto &attr = outputAttrs_[i];
        std::memset(&attr, 0, sizeof(attr));
        attr.index = static_cast<uint32_t>(i);
        RKNN_CHECK(rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)), "rknn_query_output_attr");
        dumpTensorAttr(attr);

        if (attr.type == RKNN_TENSOR_FLOAT16)
        {
            attr.type = RKNN_TENSOR_FLOAT32;
        }

        const std::size_t bytes = static_cast<std::size_t>(attr.n_elems) * tensorTypeSize(attr.type);
        auto *tensorMem = rknn_create_mem(ctx_, bytes);
        if (!tensorMem)
        {
            throw std::runtime_error("Failed to allocate RKNN output tensor memory");
        }

        outputMems_.push_back(tensorMem);
        RKNN_CHECK(rknn_set_io_mem(ctx_, tensorMem, &attr), "rknn_set_output_mem");
    }
}

void FusionNpuContext::releaseResources()
{
    for (auto *mem : inputMems_)
    {
        if (mem)
        {
            rknn_destroy_mem(ctx_, mem);
        }
    }
    inputMems_.clear();

    for (auto *mem : outputMems_)
    {
        if (mem)
        {
            rknn_destroy_mem(ctx_, mem);
        }
    }
    outputMems_.clear();

    if (initialized_)
    {
        rknn_destroy(ctx_);
        initialized_ = false;
    }
}

void FusionNpuContext::copyInputData(const cv::Mat &mat, std::size_t index)
{
    if (index >= inputAttrs_.size() || index >= inputMems_.size())
    {
        throw std::out_of_range("RKNN input index out of range");
    }

    auto &attr = inputAttrs_[index];
    auto *tensorMem = inputMems_[index];

    cv::Mat floatMat;
    if (mat.type() != CV_32FC1)
    {
        mat.convertTo(floatMat, CV_32FC1);
    }
    else
    {
        floatMat = mat;
    }

    if (!floatMat.isContinuous())
    {
        floatMat = floatMat.clone();
    }

    if (index < inputScaling_.size())
    {
        const float scaling = inputScaling_[index];
        if (std::fabs(scaling - 1.0f) > 1e-6f)
        {
            floatMat *= scaling;
        }
    }

    if (static_cast<std::size_t>(floatMat.total()) != attr.n_elems)
    {
        throw std::runtime_error("RKNN input element count mismatch");
    }

    const float *src = floatMat.ptr<float>();
    switch (attr.type)
    {
    case RKNN_TENSOR_FLOAT32:
        std::memcpy(tensorMem->virt_addr, src, static_cast<std::size_t>(attr.n_elems) * sizeof(float));
        break;
    case RKNN_TENSOR_FLOAT16:
    {
        cv::Mat halfMat;
        floatMat.convertTo(halfMat, CV_16FC1);
        if (!halfMat.isContinuous())
        {
            halfMat = halfMat.clone();
        }
        std::memcpy(tensorMem->virt_addr,
                    halfMat.ptr<std::uint16_t>(),
                    static_cast<std::size_t>(attr.n_elems) * sizeof(std::uint16_t));
        break;
    }
    case RKNN_TENSOR_INT8:
    {
        auto *dst = reinterpret_cast<std::int8_t *>(tensorMem->virt_addr);
        const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
        for (std::size_t i = 0; i < attr.n_elems; ++i)
        {
            const float quant = std::round(src[i] / scale) + static_cast<float>(attr.zp);
            dst[i] = static_cast<std::int8_t>(clampValue<int>(static_cast<int>(quant), -128, 127));
        }
        break;
    }
    case RKNN_TENSOR_UINT8:
    {
        auto *dst = reinterpret_cast<std::uint8_t *>(tensorMem->virt_addr);
        const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
        for (std::size_t i = 0; i < attr.n_elems; ++i)
        {
            const float quant = std::round(src[i] / scale) + static_cast<float>(attr.zp);
            dst[i] = static_cast<std::uint8_t>(clampValue<int>(static_cast<int>(quant), 0, 255));
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported RKNN input tensor type");
    }

    RKNN_CHECK(rknn_mem_sync(ctx_, tensorMem, RKNN_MEMORY_SYNC_TO_DEVICE), "rknn_mem_sync_input");
}

cv::Mat FusionNpuContext::fetchOutputData(std::size_t index)
{
    if (index >= outputAttrs_.size() || index >= outputMems_.size())
    {
        throw std::out_of_range("RKNN output index out of range");
    }

    auto &attr = outputAttrs_[index];
    auto *tensorMem = outputMems_[index];
    RKNN_CHECK(rknn_mem_sync(ctx_, tensorMem, RKNN_MEMORY_SYNC_FROM_DEVICE), "rknn_mem_sync_output");

    int height = 0;
    int width = 0;
    resolveSpatial(attr, height, width);
    if (height <= 0 || width <= 0)
    {
        throw std::runtime_error("Invalid RKNN output tensor shape");
    }

    cv::Mat result(height, width, CV_32FC1);
    float *dst = result.ptr<float>();

    switch (attr.type)
    {
    case RKNN_TENSOR_FLOAT32:
        std::memcpy(dst, tensorMem->virt_addr, static_cast<std::size_t>(attr.n_elems) * sizeof(float));
        break;
    case RKNN_TENSOR_INT8:
    {
        auto *src = reinterpret_cast<std::int8_t *>(tensorMem->virt_addr);
        const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
        for (std::size_t i = 0; i < attr.n_elems; ++i)
        {
            dst[i] = scale * (static_cast<std::int32_t>(src[i]) - attr.zp);
        }
        break;
    }
    case RKNN_TENSOR_UINT8:
    {
        auto *src = reinterpret_cast<std::uint8_t *>(tensorMem->virt_addr);
        const float scale = (attr.scale == 0.0f) ? 1.0f : attr.scale;
        for (std::size_t i = 0; i < attr.n_elems; ++i)
        {
            dst[i] = scale * (static_cast<std::int32_t>(src[i]) - attr.zp);
        }
        break;
    }
    default:
        throw std::runtime_error("Unsupported RKNN output tensor type");
    }

    return result;
}
