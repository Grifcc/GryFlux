#include "source/fusion_data_source.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

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

    for (const auto& vis_entry : fs::directory_iterator(vis_dir)) {
        if (!vis_entry.is_regular_file()) {
            continue;
        }

        const std::string extension = vis_entry.path().extension().string();
        if (!IsSupportedImageFile(extension)) {
            continue;
        }

        const std::string filename = vis_entry.path().filename().string();
        const fs::path ir_path = fs::path(ir_dir) / filename;
        if (!fs::exists(ir_path) || !ir_path.has_filename()) {
            LOG.warning("[FusionDataSource] Skip %s because matching IR image is missing",
                        filename.c_str());
            continue;
        }

        frame_pairs_.push_back(FramePair{
            filename,
            vis_entry.path().string(),
            ir_path.string(),
        });
    }

    std::sort(frame_pairs_.begin(), frame_pairs_.end(),
              [](const FramePair& left, const FramePair& right) {
                  return left.filename < right.filename;
              });

    setHasMore(!frame_pairs_.empty());
    if (frame_pairs_.empty()) {
        LOG.warning("[FusionDataSource] No valid frame pairs found");
    } else {
        LOG.info("[FusionDataSource] Loaded %zu frame pairs", frame_pairs_.size());
    }
}

std::unique_ptr<GryFlux::DataPacket> FusionDataSource::produce() {
    if (next_index_ >= frame_pairs_.size()) {
        setHasMore(false);
        return nullptr;
    }

    const FramePair& frame_pair = frame_pairs_[next_index_];
    cv::Mat vis_raw_bgr = cv::imread(frame_pair.vis_path, cv::IMREAD_COLOR);
    cv::Mat ir_raw_gray = cv::imread(frame_pair.ir_path, cv::IMREAD_GRAYSCALE);
    if (vis_raw_bgr.empty() || ir_raw_gray.empty()) {
        throw std::runtime_error("Failed to load fusion input pair: " + frame_pair.filename);
    }

    auto packet = std::make_unique<FusionDataPacket>(
        static_cast<uint64_t>(next_index_),
        model_width_,
        model_height_);
    packet->filename = frame_pair.filename;
    packet->vis_raw_bgr = std::move(vis_raw_bgr);
    packet->ir_raw_gray = std::move(ir_raw_gray);

    ++next_index_;
    if (next_index_ >= frame_pairs_.size()) {
        setHasMore(false);
    }
    return packet;
}

bool FusionDataSource::IsSupportedImageFile(const std::string& extension) {
    std::string lower_extension = extension;
    std::transform(lower_extension.begin(), lower_extension.end(),
                   lower_extension.begin(),
                   [](unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });

    return lower_extension == ".jpg" ||
           lower_extension == ".jpeg" ||
           lower_extension == ".png" ||
           lower_extension == ".bmp";
}
