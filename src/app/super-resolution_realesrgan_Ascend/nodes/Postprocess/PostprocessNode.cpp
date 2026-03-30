#include "PostprocessNode.h"

#include "context/realesrgan_npu_context.h"
#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include "acl/acl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace
{

struct TensorShape
{
    int n = 1;
    int c = REALESRGAN_NUM_CHANNELS;
    int h = REALESRGAN_OUTPUT_H;
    int w = REALESRGAN_OUTPUT_W;
};

std::string dimsToString(const aclmdlIODims &dims)
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

const char *formatToString(aclFormat format)
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

TensorShape finalizeShape(const TensorShape &shape, const aclmdlIODims &dims, aclFormat format, size_t elementCount)
{
    if (shape.n == 1 &&
        shape.c == REALESRGAN_NUM_CHANNELS &&
        shape.h > 0 &&
        shape.w > 0 &&
        static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w == elementCount)
    {
        return shape;
    }

    throw std::runtime_error(
        "PostprocessNode: unsupported output tensor shape. dims=" + dimsToString(dims) +
        ", format=" + std::string(formatToString(format)) +
        ", elementCount=" + std::to_string(elementCount));
}

TensorShape resolveOutputShape(const aclmdlIODims &dims, aclFormat format, size_t elementCount)
{
    TensorShape shape;

    if (dims.dimCount == 4)
    {
        if (format == ACL_FORMAT_NHWC)
        {
            shape.n = static_cast<int>(dims.dims[0]);
            shape.h = static_cast<int>(dims.dims[1]);
            shape.w = static_cast<int>(dims.dims[2]);
            shape.c = static_cast<int>(dims.dims[3]);
        }
        else
        {
            shape.n = static_cast<int>(dims.dims[0]);
            shape.c = static_cast<int>(dims.dims[1]);
            shape.h = static_cast<int>(dims.dims[2]);
            shape.w = static_cast<int>(dims.dims[3]);
        }
        return finalizeShape(shape, dims, format, elementCount);
    }

    if (dims.dimCount == 3)
    {
        shape.n = 1;
        if (format == ACL_FORMAT_NHWC)
        {
            shape.h = static_cast<int>(dims.dims[0]);
            shape.w = static_cast<int>(dims.dims[1]);
            shape.c = static_cast<int>(dims.dims[2]);
        }
        else
        {
            shape.c = static_cast<int>(dims.dims[0]);
            shape.h = static_cast<int>(dims.dims[1]);
            shape.w = static_cast<int>(dims.dims[2]);
        }
        return finalizeShape(shape, dims, format, elementCount);
    }

    const size_t expectedElementCount =
        static_cast<size_t>(REALESRGAN_NUM_CHANNELS) * REALESRGAN_OUTPUT_H * REALESRGAN_OUTPUT_W;
    if (elementCount == expectedElementCount)
    {
        return {1, REALESRGAN_NUM_CHANNELS, REALESRGAN_OUTPUT_H, REALESRGAN_OUTPUT_W};
    }

    throw std::runtime_error(
        "PostprocessNode: output tensor rank is unsupported. dims=" + dimsToString(dims) +
        ", format=" + std::string(formatToString(format)) +
        ", elementCount=" + std::to_string(elementCount));
}

uint8_t toU8FromFloat(float value)
{
    if (value <= 1.0f && value >= 0.0f)
    {
        value *= 255.0f;
    }
    value = std::clamp(value, 0.0f, 255.0f);
    return static_cast<uint8_t>(std::lround(value));
}

} // namespace

namespace PipelineNodes
{

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealEsrganPacket &>(packet);

    if (p.output_buffer.empty())
    {
        throw std::runtime_error("PostprocessNode: output_buffer is empty.");
    }

    const aclDataType outputType = p.output_data_type;
    const size_t elementSize = aclDataTypeSize(outputType);
    if (elementSize == 0)
    {
        throw std::runtime_error("PostprocessNode: unsupported output data type.");
    }

    if (p.output_buffer.size() % elementSize != 0)
    {
        throw std::runtime_error("PostprocessNode: output buffer size is not aligned to output tensor type.");
    }

    const size_t elementCount = p.output_buffer.size() / elementSize;
    const TensorShape shape = resolveOutputShape(
        p.output_dims,
        p.output_format,
        elementCount);
    p.sr_image = cv::Mat(shape.h, shape.w, CV_8UC3);

    const size_t expectedBytes = static_cast<size_t>(shape.n) * shape.c * shape.h * shape.w * elementSize;
    if (expectedBytes != p.output_buffer.size())
    {
        throw std::runtime_error("PostprocessNode: output buffer size does not match model output shape.");
    }

    for (int h = 0; h < shape.h; ++h)
    {
        auto *dst = p.sr_image.ptr<cv::Vec3b>(h);
        for (int w = 0; w < shape.w; ++w)
        {
            for (int c = 0; c < shape.c; ++c)
            {
                size_t flatIndex = 0;
                if (p.output_format == ACL_FORMAT_NHWC)
                {
                    flatIndex = static_cast<size_t>((h * shape.w + w) * shape.c + c);
                }
                else
                {
                    flatIndex = static_cast<size_t>((c * shape.h + h) * shape.w + w);
                }

                uint8_t channelValue = 0;
                if (outputType == ACL_FLOAT)
                {
                    const auto *data = reinterpret_cast<const float *>(p.output_buffer.data());
                    channelValue = toU8FromFloat(data[flatIndex]);
                }
                else if (outputType == ACL_FLOAT16)
                {
                    const auto *data = reinterpret_cast<const aclFloat16 *>(p.output_buffer.data());
                    channelValue = toU8FromFloat(aclFloat16ToFloat(data[flatIndex]));
                }
                else if (outputType == ACL_UINT8)
                {
                    const auto *data = reinterpret_cast<const uint8_t *>(p.output_buffer.data());
                    channelValue = data[flatIndex];
                }
                else if (outputType == ACL_INT8)
                {
                    const auto *data = reinterpret_cast<const int8_t *>(p.output_buffer.data());
                    channelValue = static_cast<uint8_t>(std::clamp(static_cast<int>(data[flatIndex]), 0, 255));
                }
                else
                {
                    throw std::runtime_error("PostprocessNode: unsupported output tensor type for image conversion.");
                }

                dst[w][c] = channelValue;
            }
        }
    }

    cv::cvtColor(p.sr_image, p.sr_image, cv::COLOR_RGB2BGR);

    LOG.debug("Packet %d: postprocess done, sr size=%dx%d",
              p.frame_id,
              p.sr_image.cols,
              p.sr_image.rows);

    std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
}

} // namespace PipelineNodes
