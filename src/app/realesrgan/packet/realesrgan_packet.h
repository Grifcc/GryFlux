#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>

struct RealesrganPacket : public GryFlux::DataPacket
{
    int idx = 0;
    std::string filename;

    cv::Mat inputBgrU8;
    cv::Mat modelRgbU8;
    cv::Mat srTensorF32;
    cv::Mat outputBgrU8;

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(idx);
    }
};
