#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ZeroDcePacket : public GryFlux::DataPacket {
    uint64_t frame_id = 0;
    std::string image_path;
    std::string image_name;
    std::string source_filename;
    std::string gt_path;
    bool has_ground_truth = false;
    bool enable_save = true;
    bool enable_metrics = true;
    bool infer_only = false;

    int input_channels = 3;
    int input_height = 0;
    int input_width = 0;
    int output_channels = 3;
    int output_height = 0;
    int output_width = 0;

    cv::Mat input_image;
    cv::Mat output_image;

    std::vector<float> input_tensor;
    std::vector<float> output_tensor;

    bool is_valid_image = true;
    bool write_enqueued = false;
    std::string status = "pending";
    std::string error_message;

    double input_mean_luma = 0.0;
    double output_mean_luma = 0.0;
    double int8_psnr = 0.0;
    double loss = 0.0;
    bool is_proxy_psnr = true;
    double preprocess_ms = 0.0;
    double infer_ms = 0.0;
    double postprocess_ms = 0.0;

    static size_t ElementCount(int channels, int height, int width) {
        return static_cast<size_t>(channels) * static_cast<size_t>(height) * static_cast<size_t>(width);
    }

    ZeroDcePacket(int input_channels_arg,
                  int input_height_arg,
                  int input_width_arg,
                  int output_channels_arg,
                  int output_height_arg,
                  int output_width_arg)
        : input_channels(input_channels_arg),
          input_height(input_height_arg),
          input_width(input_width_arg),
          output_channels(output_channels_arg),
          output_height(output_height_arg),
          output_width(output_width_arg),
          input_tensor(ElementCount(input_channels_arg, input_height_arg, input_width_arg), 0.0f),
          output_tensor(ElementCount(output_channels_arg, output_height_arg, output_width_arg), 0.0f) {}

    uint64_t getIdx() const override { return frame_id; }
};
