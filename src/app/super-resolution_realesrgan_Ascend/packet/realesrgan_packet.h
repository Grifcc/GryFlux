#pragma once

#include "framework/data_packet.h"

#include "acl/acl.h"

#include <opencv2/opencv.hpp>

#include <limits>
#include <string>
#include <cstdint>
#include <vector>

constexpr int REALESRGAN_INPUT_W = 256;
constexpr int REALESRGAN_INPUT_H = 256;
constexpr int REALESRGAN_SCALE_FACTOR = 4;
constexpr int REALESRGAN_OUTPUT_W = REALESRGAN_INPUT_W * REALESRGAN_SCALE_FACTOR;
constexpr int REALESRGAN_OUTPUT_H = REALESRGAN_INPUT_H * REALESRGAN_SCALE_FACTOR;
constexpr int REALESRGAN_NUM_CHANNELS = 3;

struct RealEsrganPacket : public GryFlux::DataPacket
{
    int frame_id = 0;
    std::string lr_path;
    std::string hr_path;

    int lr_w = 0;
    int lr_h = 0;

    cv::Mat lr_image;
    cv::Mat hr_image;
    cv::Mat sr_image;

    std::vector<float> input_tensor;
    std::vector<uint8_t> output_buffer;
    aclmdlIODims output_dims{};
    aclFormat output_format = ACL_FORMAT_UNDEFINED;
    aclDataType output_data_type = ACL_DT_UNDEFINED;
    double psnr = std::numeric_limits<double>::quiet_NaN();
    bool has_valid_psnr = false;

    RealEsrganPacket()
        : input_tensor(REALESRGAN_NUM_CHANNELS * REALESRGAN_INPUT_H * REALESRGAN_INPUT_W)
    {
    }

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(frame_id);
    }
};
