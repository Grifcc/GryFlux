#pragma once

#include <string>
#include <opencv2/opencv.hpp>
#include "framework/data_packet.h"

inline int g_fusion_model_width = 640;
inline int g_fusion_model_height = 480;

inline void SetFusionModelSize(int width, int height) {
    if (width > 0) {
        g_fusion_model_width = width;
    }
    if (height > 0) {
        g_fusion_model_height = height;
    }
}

inline int GetFusionModelWidth() {
    return g_fusion_model_width;
}

inline int GetFusionModelHeight() {
    return g_fusion_model_height;
}

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
        const int model_width = GetFusionModelWidth();
        const int model_height = GetFusionModelHeight();

        vis_resize.create(model_height, model_width, CV_8UC3);
        ir_resize.create(model_height, model_width, CV_8UC1);
        ycrcb_img.create(model_height, model_width, CV_8UC3);
        
        vis_y.create(model_height, model_width, CV_8UC1);
        vis_cr.create(model_height, model_width, CV_8UC1);
        vis_cb.create(model_height, model_width, CV_8UC1);

        vis_y_float.create(model_height, model_width, CV_32FC1);
        ir_float.create(model_height, model_width, CV_32FC1);

        fused_y_float.create(model_height, model_width, CV_32FC1);
        fused_y_uint8.create(model_height, model_width, CV_8UC1);
        fused_ycrcb.create(model_height, model_width, CV_8UC3);
        fused_result.create(model_height, model_width, CV_8UC3);
    }

    ~FusionDataPacket() override = default;

    // 实现基类的纯虚函数，让框架能够追踪 Packet
    uint64_t getIdx() const override {
        return packet_idx;
    }
};
