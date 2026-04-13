#include "source/image_data_source.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <memory>
#include <stdexcept>

ImageDataSource::ImageDataSource(
    const std::string& video_path,
    int detection_model_width,
    int detection_model_height,
    int reid_width,
    int reid_height,
    int reid_feature_dim)
    : cap_(video_path),
      detection_model_width_(detection_model_width),
      detection_model_height_(detection_model_height),
      reid_width_(reid_width),
      reid_height_(reid_height),
      reid_feature_dim_(reid_feature_dim) {
    if (!cap_.isOpened()) {
        throw std::runtime_error("Failed to open input video: " + video_path);
    }

    setHasMore(true);
    ReadNextFrame();
    LOG.info("[ImageDataSource] Opened %s, fps=%.2f", video_path.c_str(), getFps());
}

ImageDataSource::~ImageDataSource() {
    if (cap_.isOpened()) {
        cap_.release();
    }
}

std::unique_ptr<GryFlux::DataPacket> ImageDataSource::produce() {
    if (!hasMore()) {
        return nullptr;
    }

    auto packet = std::make_unique<TrackDataPacket>(
        detection_model_width_,
        detection_model_height_,
        reid_width_,
        reid_height_,
        reid_feature_dim_);
    packet->original_image = next_frame_.clone();
    packet->frame_id = frame_count_++;

    ReadNextFrame();
    return packet;
}

void ImageDataSource::ReadNextFrame() {
    cap_ >> next_frame_;
    if (!next_frame_.empty()) {
        return;
    }

    setHasMore(false);
    LOG.info("[ImageDataSource] Completed after %d frames", frame_count_);
}
