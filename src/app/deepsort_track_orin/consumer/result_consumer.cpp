#include "result_consumer.h"

#include "utils/logger.h"

#include <Eigen/Core>

#include <stdexcept>

namespace {

double NormalizeFps(double fps) {
    return fps > 0.0 ? fps : 25.0;
}

}  // namespace

ResultConsumer::ResultConsumer(
    const std::string& output_path,
    double fps,
    int width,
    int height)
    : tracker_(std::make_unique<DeepSortTracker>(0.4f, 100)),
      writer_(output_path,
              cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
              NormalizeFps(fps),
              cv::Size(width, height)) {
    if (!writer_.isOpened()) {
        throw std::runtime_error("Failed to create tracking output video: " + output_path);
    }

    LOG.info("[ResultConsumer] Writing DeepSORT output to %s", output_path.c_str());
}

ResultConsumer::~ResultConsumer() {
    if (writer_.isOpened()) {
        writer_.release();
        LOG.info("[ResultConsumer] Output video finalized");
    }
}

void ResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* track_packet = static_cast<TrackDataPacket*>(packet.get());
    reorder_buffer_[track_packet->frame_id] = std::move(packet);

    while (reorder_buffer_.count(expected_frame_id_) > 0) {
        auto current_packet = std::move(reorder_buffer_[expected_frame_id_]);
        reorder_buffer_.erase(expected_frame_id_);
        ProcessSequentialFrame(static_cast<TrackDataPacket*>(current_packet.get()));
        ++expected_frame_id_;
    }
}

void ResultConsumer::ProcessSequentialFrame(TrackDataPacket* packet) {
    DETECTIONS tracker_input;
    const size_t usable_count =
        std::min(packet->detections.size(), packet->active_reid_feature_count);
    tracker_input.reserve(usable_count);

    if (packet->detections.size() != packet->active_reid_feature_count) {
        LOG.warning(
            "[ResultConsumer] Frame %d detection/feature count mismatch: %zu vs %zu",
            packet->frame_id,
            packet->detections.size(),
            packet->active_reid_feature_count);
    }

    for (size_t index = 0; index < usable_count; ++index) {
        const auto& detection = packet->detections[index];
        const auto& feature_data = packet->reid_features[index];
        if (feature_data.empty()) {
            continue;
        }

        DETECTBOX box;
        box << detection.x1,
               detection.y1,
               detection.x2 - detection.x1,
               detection.y2 - detection.y1;

        FEATURE feature = Eigen::Map<const FEATURE>(feature_data.data());
        tracker_input.emplace_back(box, detection.score, feature);
    }

    packet->active_tracks = tracker_->update(tracker_input);

    for (const auto& track : packet->active_tracks) {
        const auto tlwh = track.to_tlwh();
        const cv::Rect rect(tlwh(0), tlwh(1), tlwh(2), tlwh(3));
        cv::rectangle(packet->original_image, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(packet->original_image,
                    "ID: " + std::to_string(track.track_id),
                    cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2);
    }

    writer_.write(packet->original_image);
    if (packet->frame_id % 30 == 0) {
        LOG.info("[ResultConsumer] Processed frame %d", packet->frame_id);
    }
}
