#include "PreprocessNode.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>

namespace {

double ComputeMeanLuma(const cv::Mat& bgr_image) {
    cv::Mat gray;
    cv::cvtColor(bgr_image, gray, cv::COLOR_BGR2GRAY);
    return cv::mean(gray)[0];
}

}  // namespace

void PreprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& dce_packet = static_cast<ZeroDcePacket&>(packet);
    (void)ctx;
    const auto start_time = std::chrono::steady_clock::now();

    if (dce_packet.input_image.empty()) {
        dce_packet.is_valid_image = false;
        dce_packet.status = "decode_failed";
        dce_packet.error_message = "image decode failed";
        std::fill(dce_packet.input_tensor.begin(), dce_packet.input_tensor.end(), 0.0f);
        const auto end_time = std::chrono::steady_clock::now();
        dce_packet.preprocess_ms =
            std::chrono::duration<double, std::milli>(end_time - start_time).count();
        return;
    }

    dce_packet.input_mean_luma = ComputeMeanLuma(dce_packet.input_image);
    dce_packet.is_valid_image = true;
    dce_packet.status = "preprocessed";
    dce_packet.error_message.clear();

    cv::Mat resized;
    cv::resize(
        dce_packet.input_image,
        resized,
        cv::Size(dce_packet.input_width, dce_packet.input_height),
        0.0,
        0.0,
        cv::INTER_LINEAR);

    cv::Mat rgb_image;
    cv::cvtColor(resized, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat rgb_float;
    rgb_image.convertTo(rgb_float, CV_32FC3, 1.0 / 255.0);

    const int image_area = dce_packet.input_width * dce_packet.input_height;
    std::vector<cv::Mat> chw_channels;
    chw_channels.reserve(3);
    chw_channels.emplace_back(
        dce_packet.input_height, dce_packet.input_width, CV_32FC1, dce_packet.input_tensor.data());
    chw_channels.emplace_back(
        dce_packet.input_height, dce_packet.input_width, CV_32FC1, dce_packet.input_tensor.data() + image_area);
    chw_channels.emplace_back(
        dce_packet.input_height,
        dce_packet.input_width,
        CV_32FC1,
        dce_packet.input_tensor.data() + (2 * image_area));
    cv::split(rgb_float, chw_channels);

    const auto end_time = std::chrono::steady_clock::now();
    dce_packet.preprocess_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
}
