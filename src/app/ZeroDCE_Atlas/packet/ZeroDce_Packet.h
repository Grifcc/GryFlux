#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "framework/data_packet.h"

struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id;
    
    uint64_t getIdx() const override { return frame_id; }
    
    cv::Mat input_image;
    cv::Mat output_image;
    std::vector<float> input_tensor;
    std::vector<unsigned char> host_output_buffer;

    std::string image_name; 
    double int8_psnr = 0.0;
    double loss = 0.0;
    std::string status = "OK";
    
};
using ZeroDcePacketPtr = std::shared_ptr<ZeroDcePacket>;
