#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>

namespace PipelineNodes {

PreprocessNode::PreprocessNode(int model_width, int model_height)
    : model_width_(model_width),
      model_height_(model_height) {}

void PreprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& detect_packet = static_cast<DetectDataPacket&>(packet);
    (void)ctx;

    if (detect_packet.original_image.empty()) {
        LOG.error("[PreprocessNode] Packet %d has an empty input image",
                  detect_packet.frame_id);
        detect_packet.markFailed();
        return;
    }

    cv::Mat rgb_image;
    cv::cvtColor(detect_packet.original_image, rgb_image, cv::COLOR_BGR2RGB);

    const int image_width = rgb_image.cols;
    const int image_height = rgb_image.rows;
    detect_packet.preproc_data.original_width = image_width;
    detect_packet.preproc_data.original_height = image_height;

    cv::Mat processed_float_image;
    if (image_width == model_width_ && image_height == model_height_) {
        rgb_image.convertTo(processed_float_image, CV_32FC3);
        detect_packet.preproc_data.scale = 1.0f;
        detect_packet.preproc_data.x_offset = 0;
        detect_packet.preproc_data.y_offset = 0;
    } else {
        const float scale = std::min(
            static_cast<float>(model_width_) / static_cast<float>(image_width),
            static_cast<float>(model_height_) / static_cast<float>(image_height));
        detect_packet.preproc_data.scale = scale;

        const int resized_width = static_cast<int>(image_width * scale);
        const int resized_height = static_cast<int>(image_height * scale);

        cv::Mat resized_image;
        cv::resize(
            rgb_image,
            resized_image,
            cv::Size(resized_width, resized_height));

        cv::Mat letterbox_image(
            model_height_,
            model_width_,
            CV_8UC3,
            cv::Scalar(114, 114, 114));

        detect_packet.preproc_data.x_offset =
            (model_width_ - resized_width) / 2;
        detect_packet.preproc_data.y_offset =
            (model_height_ - resized_height) / 2;

        resized_image.copyTo(
            letterbox_image(cv::Rect(
                detect_packet.preproc_data.x_offset,
                detect_packet.preproc_data.y_offset,
                resized_width,
                resized_height)));
        letterbox_image.convertTo(processed_float_image, CV_32FC3);
    }

    const int plane_size = model_width_ * model_height_;
    float* nchw_buffer = detect_packet.preproc_data.nchw_data.data();

    if (processed_float_image.isContinuous()) {
        const float* hwc_buffer = processed_float_image.ptr<float>();
        for (int index = 0; index < plane_size; ++index) {
            nchw_buffer[index] = hwc_buffer[index * 3 + 0];
            nchw_buffer[plane_size + index] = hwc_buffer[index * 3 + 1];
            nchw_buffer[plane_size * 2 + index] = hwc_buffer[index * 3 + 2];
        }
        return;
    }

    for (int height = 0; height < model_height_; ++height) {
        for (int width = 0; width < model_width_; ++width) {
            const float* pixel = processed_float_image.ptr<float>(height, width);
            const int base_index = height * model_width_ + width;
            nchw_buffer[base_index] = pixel[0];
            nchw_buffer[plane_size + base_index] = pixel[1];
            nchw_buffer[plane_size * 2 + base_index] = pixel[2];
        }
    }
}

}  // namespace PipelineNodes
