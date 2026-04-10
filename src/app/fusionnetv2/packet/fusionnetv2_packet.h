#pragma once

#include "framework/data_packet.h"

#include <opencv2/core.hpp>

#include <cstdint>
#include <string>

struct FusionNetV2Packet : public GryFlux::DataPacket
{
    int idx = 0;
    std::string filename;

    cv::Mat visibleBgrU8;
    cv::Mat infraredGrayU8;

    cv::Mat visYF32;
    cv::Mat visCbU8;
    cv::Mat visCrU8;
    cv::Mat infraredF32;
    cv::Size originalVisibleSize;

    cv::Mat fusedYF32;
    cv::Mat outputBgrU8;

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(idx);
    }
};
