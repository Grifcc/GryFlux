#include "preprocess.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);

    if (fusion_packet.vis_raw.empty() || fusion_packet.ir_raw.empty()) {
        std::cerr << "[PreprocessNode] Error: Raw images are empty. File: " 
                  << fusion_packet.filename << std::endl;
        return;
    }

    cv::resize(fusion_packet.vis_raw, fusion_packet.vis_resize, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    cv::resize(fusion_packet.ir_raw, fusion_packet.ir_resize, cv::Size(MODEL_WIDTH, MODEL_HEIGHT));
    cv::cvtColor(fusion_packet.vis_resize, fusion_packet.ycrcb_img, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels = {
        fusion_packet.vis_y, 
        fusion_packet.vis_cr, 
        fusion_packet.vis_cb
    };
    cv::split(fusion_packet.ycrcb_img, channels);

    fusion_packet.vis_y.convertTo(fusion_packet.vis_y_float, CV_32FC1, 1.0 / 255.0);
    fusion_packet.ir_resize.convertTo(fusion_packet.ir_float, CV_32FC1, 1.0 / 255.0);
}