#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "framework/data_packet.h"

constexpr int MODEL_WIDTH = 640;
constexpr int MODEL_HEIGHT = 480;

class FusionDataPacket : public GryFlux::DataPacket {
public:
    uint64_t packet_idx = 0;  // 添加序列号字段
    std::string filename;

    cv::Mat vis_raw;
    cv::Mat ir_raw;

    cv::Mat vis_resize;    
    cv::Mat ir_resize;     
    cv::Mat ycrcb_img;     
    
    cv::Mat vis_y;         
    cv::Mat vis_cr;        
    cv::Mat vis_cb;        

    cv::Mat vis_y_float;   
    cv::Mat ir_float;      

    cv::Mat fused_y_float; 
    cv::Mat fused_y_uint8; 
    cv::Mat fused_ycrcb;   
    cv::Mat fused_result;  

    FusionDataPacket() {
        vis_resize.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
        ir_resize.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC1);
        ycrcb_img.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
        
        vis_y.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC1);
        vis_cr.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC1);
        vis_cb.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC1);

        vis_y_float.create(MODEL_HEIGHT, MODEL_WIDTH, CV_32FC1);
        ir_float.create(MODEL_HEIGHT, MODEL_WIDTH, CV_32FC1);

        fused_y_float.create(MODEL_HEIGHT, MODEL_WIDTH, CV_32FC1);
        fused_y_uint8.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC1);
        fused_ycrcb.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
        fused_result.create(MODEL_HEIGHT, MODEL_WIDTH, CV_8UC3);
    }

    ~FusionDataPacket() override = default;

    // 实现基类的纯虚函数，让框架能够追踪 Packet
    uint64_t getIdx() const override {
        return packet_idx;
    }
};