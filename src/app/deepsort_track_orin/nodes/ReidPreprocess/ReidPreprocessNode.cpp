#include "nodes/ReidPreprocess/ReidPreprocessNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>

namespace PipelineNodes {

ReidPreprocessNode::ReidPreprocessNode(int target_width, int target_height)
    : target_width_(target_width),
      target_height_(target_height) {}

void ReidPreprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    (void)ctx;

    if (track_packet.detections.empty()) {
        return;
    }
    const size_t crop_capacity = track_packet.reid_preproc_crops.size();
    track_packet.active_reid_crop_count =
        std::min(track_packet.detections.size(), crop_capacity);
    if (track_packet.detections.size() > crop_capacity) {
        LOG.warning(
            "[ReidPreprocessNode] Packet %d has %zu detections but only %zu ReID slots",
            track_packet.frame_id,
            track_packet.detections.size(),
            crop_capacity);
    }

    for (size_t index = 0; index < track_packet.active_reid_crop_count; ++index) {
        const auto& detection = track_packet.detections[index];
        const int x = std::max(0, static_cast<int>(detection.x1));
        const int y = std::max(0, static_cast<int>(detection.y1));
        const int width = std::min(
            track_packet.original_image.cols - x,
            static_cast<int>(detection.x2 - detection.x1));
        const int height = std::min(
            track_packet.original_image.rows - y,
            static_cast<int>(detection.y2 - detection.y1));

        auto& crop_buffer = track_packet.reid_preproc_crops[index];
        if (width <= 0 || height <= 0) {
            std::fill(crop_buffer.begin(), crop_buffer.end(), 0.0f);
            track_packet.reid_crop_valid_flags[index] = 0U;
            continue;
        }

        const cv::Mat crop =
            track_packet.original_image(cv::Rect(x, y, width, height));

        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(target_width_, target_height_));
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        std::vector<cv::Mat> channels(3);
        for (int channel = 0; channel < 3; ++channel) {
            channels[channel] = cv::Mat(
                target_height_,
                target_width_,
                CV_32FC1,
                crop_buffer.data() +
                    static_cast<size_t>(channel) * target_width_ * target_height_);
        }
        cv::split(resized, channels);
        track_packet.reid_crop_valid_flags[index] = 1U;
    }
}

}  // namespace PipelineNodes
