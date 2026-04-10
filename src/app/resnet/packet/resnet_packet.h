#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ResnetPacket : public GryFlux::DataPacket
{
    struct TopKResult
    {
        int classId = -1;
        float probability = 0.0f;
        std::string label;
    };

    int idx = 0;
    std::string imagePath;

    cv::Mat originalImage;
    cv::Mat preprocessedImage;
    std::vector<uint8_t> inputData;

    std::size_t modelWidth = 224;
    std::size_t modelHeight = 224;

    std::vector<float> logits;
    std::vector<std::size_t> sortedIndices;
    std::vector<TopKResult> topK;

    ResnetPacket()
        : inputData(224 * 224 * 3),
          logits(1000),
          sortedIndices(1000)
    {
        topK.reserve(5);
    }

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(idx);
    }
};

