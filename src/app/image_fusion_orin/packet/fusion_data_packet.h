#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <string>

class FusionDataPacket : public GryFlux::DataPacket {
public:
    FusionDataPacket(uint64_t packet_index, int model_width, int model_height)
        : packet_idx(packet_index),
          model_width_(model_width),
          model_height_(model_height) {
        vis_resized_bgr.create(model_height_, model_width_, CV_8UC3);
        ir_resized_gray.create(model_height_, model_width_, CV_8UC1);
        vis_ycrcb.create(model_height_, model_width_, CV_8UC3);
        vis_y.create(model_height_, model_width_, CV_8UC1);
        vis_cr.create(model_height_, model_width_, CV_8UC1);
        vis_cb.create(model_height_, model_width_, CV_8UC1);
        vis_y_float.create(model_height_, model_width_, CV_32FC1);
        ir_float.create(model_height_, model_width_, CV_32FC1);
        fused_y_float.create(model_height_, model_width_, CV_32FC1);
        fused_y_uint8.create(model_height_, model_width_, CV_8UC1);
        fused_ycrcb.create(model_height_, model_width_, CV_8UC3);
        fused_result.create(model_height_, model_width_, CV_8UC3);
    }

    ~FusionDataPacket() override = default;

    uint64_t getIdx() const override { return packet_idx; }

    int modelWidth() const { return model_width_; }
    int modelHeight() const { return model_height_; }

    uint64_t packet_idx = 0;
    std::string filename;

    cv::Mat vis_raw_bgr;
    cv::Mat ir_raw_gray;

    cv::Mat vis_resized_bgr;
    cv::Mat ir_resized_gray;
    cv::Mat vis_ycrcb;

    cv::Mat vis_y;
    cv::Mat vis_cr;
    cv::Mat vis_cb;

    cv::Mat vis_y_float;
    cv::Mat ir_float;

    cv::Mat fused_y_float;
    cv::Mat fused_y_uint8;
    cv::Mat fused_ycrcb;
    cv::Mat fused_result;

private:
    int model_width_ = 0;
    int model_height_ = 0;
};
