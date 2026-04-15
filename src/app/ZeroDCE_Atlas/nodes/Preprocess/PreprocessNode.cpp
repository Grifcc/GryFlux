#include "PreprocessNode.h"
#include <opencv2/opencv.hpp>

namespace {

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 480;

}

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);
    (void)ctx;

    if (dce_packet.input_image.empty()) {
        throw std::runtime_error("input image is empty");
    }

    cv::Mat resized_image;
    cv::resize(dce_packet.input_image, resized_image, cv::Size(kInputWidth, kInputHeight));

    cv::Mat rgb_image;
    cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);

    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

    dce_packet.input_tensor.resize(3 * kInputHeight * kInputWidth);

    size_t channel_size = static_cast<size_t>(kInputHeight) * kInputWidth;
    for (int y = 0; y < kInputHeight; ++y) {
        for (int x = 0; x < kInputWidth; ++x) {
            const cv::Vec3f& pixel = float_image.at<cv::Vec3f>(y, x);
            size_t offset = static_cast<size_t>(y) * kInputWidth + x;
            dce_packet.input_tensor[offset] = pixel[0];
            dce_packet.input_tensor[channel_size + offset] = pixel[1];
            dce_packet.input_tensor[channel_size * 2 + offset] = pixel[2];
        }
    }
}
