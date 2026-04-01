#pragma once
#include "framework/async_pipeline.h"
#include <vector>
#include <string>

struct ResNetPacket : public GryFlux::DataPacket {
    uint64_t packet_id = 0; 

    std::string image_path;
    int ground_truth_label = -1;
    std::vector<float> preprocessed_data;
    std::vector<float> logits;
    int top1_class = -1;
    bool top5_correct = false;

    ResNetPacket() 
        : preprocessed_data(3 * 224 * 224), 
          logits(1000) 
    {}

    uint64_t getIdx() const override {
        return packet_id;
    }
};