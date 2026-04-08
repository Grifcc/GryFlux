#pragma once
#include "framework/data_packet.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct PreprocessData {
    std::vector<float> nchw_data; 
    
    float scale = 1.0f;
    int x_offset = 0;
    int y_offset = 0;
    int original_width = 0;
    int original_height = 0;
};

struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};

struct DetectDataPacket : public GryFlux::DataPacket {
    int frame_id = 0;
    cv::Mat original_image;
    
    PreprocessData preproc_data;
    std::vector<std::vector<float>> infer_outputs;
    std::vector<Detection> detections;

    DetectDataPacket() {
        infer_outputs.resize(9);
        detections.reserve(100);
        preproc_data.nchw_data.resize(3 * 640 * 640); 
    }

    uint64_t getIdx() const override {
        return static_cast<uint64_t>(frame_id);
    }
};