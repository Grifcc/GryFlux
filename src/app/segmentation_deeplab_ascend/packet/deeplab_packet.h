/*************************************************************************************************************************
 * Copyright 2025 FallenSoul-He
 *
 * GryFlux Framework - Simple Data Packet
 *************************************************************************************************************************/
#pragma once

#include "framework/data_packet.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

constexpr int MODEL_INPUT_W = 513;
constexpr int MODEL_INPUT_H = 513;
constexpr int MODEL_OUT_W = 65;
constexpr int MODEL_OUT_H = 65;
constexpr int NUM_CLASSES = 21;

/**
 * @brief DeepLabV3 数据包
 */
struct DeepLabPacket : public GryFlux::DataPacket {
    int frame_id = 0;
    std::string image_path;
    int orig_w = 0;
    int orig_h = 0;
    std::vector<float> input_tensor;
    std::vector<float> output_tensor;
    cv::Mat pred_mask_resized;
    DeepLabPacket()
        : input_tensor(3 * MODEL_INPUT_H * MODEL_INPUT_W),
          output_tensor(MODEL_OUT_H * MODEL_OUT_W * NUM_CLASSES)
    {
    }

    uint64_t getIdx() const override {
        return static_cast<uint64_t>(frame_id);
    }
};
