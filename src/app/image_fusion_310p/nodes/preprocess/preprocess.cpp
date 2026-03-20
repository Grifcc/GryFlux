#include "preprocess.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);

    if (fusion_packet.vis_raw.empty() || fusion_packet.ir_raw.empty()) {
        std::cerr << "[PreprocessNode] Error: Raw images are empty. File: " 
                  << fusion_packet.filename << std::endl;
        return;
    }

    // 1. 尺寸调整 (Resize)
    cv::resize(fusion_packet.vis_raw, fusion_packet.vis_resize, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    cv::resize(fusion_packet.ir_raw, fusion_packet.ir_resize, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    
    // 2. 颜色空间转换 (BGR -> YCrCb)
    cv::cvtColor(fusion_packet.vis_resize, fusion_packet.ycrcb_img, cv::COLOR_BGR2YCrCb);

    // 3. 【关键修复】通道分离：使用 extractChannel 精准抽取
    // 索引 0=Y, 1=Cr, 2=Cb
    cv::extractChannel(fusion_packet.ycrcb_img, fusion_packet.vis_y, 0);  
    cv::extractChannel(fusion_packet.ycrcb_img, fusion_packet.vis_cr, 1); 
    cv::extractChannel(fusion_packet.ycrcb_img, fusion_packet.vis_cb, 2); 

    // 4. 归一化与类型转换
    fusion_packet.vis_y.convertTo(fusion_packet.vis_y_float, CV_32FC1, 1.0 / 255.0);
    fusion_packet.ir_resize.convertTo(fusion_packet.ir_float, CV_32FC1, 1.0 / 255.0);
}