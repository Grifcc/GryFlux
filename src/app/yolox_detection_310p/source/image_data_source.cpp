#include "source/image_data_source.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

#include <memory>
#include <stdexcept>
#include <string>

ImageDataSource::ImageDataSource(
    const std::string& video_path,
    int model_width,
    int model_height)
    : cap_(video_path),
      model_width_(model_width),
      model_height_(model_height) {
    if (!cap_.isOpened()) {
        throw std::runtime_error("Failed to open input video: " + video_path);
    }

    setHasMore(true);
    ReadNextFrame();
    LOG.info("[ImageDataSource] Opened %s, fps=%.2f",
             video_path.c_str(),
             getFps());
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

    auto packet = std::make_unique<DetectDataPacket>(model_width_, model_height_);
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
