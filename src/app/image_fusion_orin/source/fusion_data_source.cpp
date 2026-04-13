#include "source/fusion_data_source.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <utility>

namespace fs = std::filesystem;

FusionDataSource::FusionDataSource(
    const std::string& vis_dir,
    const std::string& ir_dir,
    int model_width,
    int model_height)
    : model_width_(model_width),
      model_height_(model_height) {
    if (!fs::exists(vis_dir) || !fs::is_directory(vis_dir)) {
        throw std::runtime_error("Visible image directory does not exist: " + vis_dir);
    }
    if (!fs::exists(ir_dir) || !fs::is_directory(ir_dir)) {
        throw std::runtime_error("Infrared image directory does not exist: " + ir_dir);
    }

    for (const auto& entry : fs::directory_iterator(vis_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string extension = entry.path().extension().string();
        if (!IsSupportedImageFile(extension)) {
            continue;
        }

        const fs::path ir_path = fs::path(ir_dir) / entry.path().filename();
        if (!fs::exists(ir_path) || !fs::is_regular_file(ir_path)) {
            LOG.warning(
                "[FusionDataSource] Skip %s because paired IR image is missing",
                entry.path().filename().string().c_str());
            continue;
        }

        frame_pairs_.push_back(
            FramePair{
                entry.path().filename().string(),
                entry.path().string(),
                ir_path.string()});
    }

    std::sort(
        frame_pairs_.begin(),
        frame_pairs_.end(),
        [](const FramePair& lhs, const FramePair& rhs) {
            return lhs.filename < rhs.filename;
        });

    setHasMore(!frame_pairs_.empty());
    LOG.info("[FusionDataSource] Found %zu paired images", frame_pairs_.size());
}

std::unique_ptr<GryFlux::DataPacket> FusionDataSource::produce() {
    while (next_index_ < frame_pairs_.size()) {
        const FramePair& frame_pair = frame_pairs_[next_index_++];

        auto packet = std::make_unique<FusionDataPacket>(
            next_index_ - 1,
            model_width_,
            model_height_);
        packet->filename = frame_pair.filename;
        packet->vis_raw_bgr = cv::imread(frame_pair.vis_path, cv::IMREAD_COLOR);
        packet->ir_raw_gray = cv::imread(frame_pair.ir_path, cv::IMREAD_GRAYSCALE);

        if (packet->vis_raw_bgr.empty() || packet->ir_raw_gray.empty()) {
            LOG.warning(
                "[FusionDataSource] Failed to read image pair %s, skipping",
                frame_pair.filename.c_str());
            continue;
        }

        setHasMore(next_index_ < frame_pairs_.size());
        return packet;
    }

    setHasMore(false);
    return nullptr;
}

bool FusionDataSource::IsSupportedImageFile(const std::string& extension) {
    std::string lowered_extension = extension;
    std::transform(
        lowered_extension.begin(),
        lowered_extension.end(),
        lowered_extension.begin(),
        [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
    return lowered_extension == ".jpg" ||
           lowered_extension == ".jpeg" ||
           lowered_extension == ".png" ||
           lowered_extension == ".bmp" ||
           lowered_extension == ".tif" ||
           lowered_extension == ".tiff";
}
