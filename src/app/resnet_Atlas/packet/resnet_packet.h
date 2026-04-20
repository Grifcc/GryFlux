#pragma once

#include "framework/data_packet.h"

#include <cstdint>
#include <string>
#include <vector>

struct ResNetPacket : public GryFlux::DataPacket {
    static constexpr int kNumClasses = 1000;
    static constexpr int kInputChannels = 3;
    static constexpr int kInputHeight = 224;
    static constexpr int kInputWidth = 224;

    uint64_t packet_id = 0;

    std::string image_path;
    int ground_truth_label = -1;

    std::vector<float> preprocessed_data;
    std::vector<float> logits;

    int top1_class = -1;
    bool top5_correct = false;

    ResNetPacket()
        : preprocessed_data(
              static_cast<size_t>(kInputChannels) *
              static_cast<size_t>(kInputHeight) *
              static_cast<size_t>(kInputWidth)),
          logits(static_cast<size_t>(kNumClasses)) {}

    uint64_t getIdx() const override { return packet_id; }
};
