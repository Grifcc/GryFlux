#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <cmath>

namespace {

constexpr int kImgWidth = 224;
constexpr int kImgHeight = 224;
constexpr int kResizeShortEdge = 256;
constexpr float kMeanRgb[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStdRgb[3]  = {0.229f, 0.224f, 0.225f};

}  // namespace

namespace PipelineNodes {

void PreprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& resnet_packet = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    cv::Mat image = cv::imread(resnet_packet.image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        LOG.error("[PreprocessNode] Packet %llu: image missing or corrupt: %s",
                  static_cast<unsigned long long>(resnet_packet.packet_id),
                  resnet_packet.image_path.c_str());
        std::fill(resnet_packet.preprocessed_data.begin(),
                  resnet_packet.preprocessed_data.end(),
                  0.0f);
        resnet_packet.markFailed();
        return;
    }

    const int h = image.rows;
    const int w = image.cols;
    int new_w = 0;
    int new_h = 0;
    if (h < w) {
        new_h = kResizeShortEdge;
        new_w = static_cast<int>(std::round(w * (kResizeShortEdge / static_cast<double>(h))));
    } else {
        new_w = kResizeShortEdge;
        new_h = static_cast<int>(std::round(h * (kResizeShortEdge / static_cast<double>(w))));
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_w, new_h));

    const int x = (new_w - kImgWidth) / 2;
    const int y = (new_h - kImgHeight) / 2;
    cv::Mat cropped_image = resized_image(cv::Rect(x, y, kImgWidth, kImgHeight)).clone();

    const int num_pixels = kImgWidth * kImgHeight;
    float* ptr_r = &resnet_packet.preprocessed_data[0];
    float* ptr_g = &resnet_packet.preprocessed_data[num_pixels];
    float* ptr_b = &resnet_packet.preprocessed_data[num_pixels * 2];

    const uint8_t* pixel_ptr = cropped_image.data;
    for (int i = 0; i < num_pixels; ++i) {
        const float b = static_cast<float>(pixel_ptr[0]) / 255.0f;
        const float g = static_cast<float>(pixel_ptr[1]) / 255.0f;
        const float r = static_cast<float>(pixel_ptr[2]) / 255.0f;
        pixel_ptr += 3;

        *ptr_r++ = (r - kMeanRgb[0]) / kStdRgb[0];
        *ptr_g++ = (g - kMeanRgb[1]) / kStdRgb[1];
        *ptr_b++ = (b - kMeanRgb[2]) / kStdRgb[2];
    }
}

}  // namespace PipelineNodes
