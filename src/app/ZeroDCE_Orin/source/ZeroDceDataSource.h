#pragma once

#include "../packet/zero_dce_packet.h"
#include "framework/data_source.h"

#include <filesystem>
#include <unordered_map>
#include <string>
#include <vector>

class ZeroDceDataSource : public GryFlux::DataSource {
public:
    ZeroDceDataSource(const std::string& input_dir,
                      const std::string& gt_dir,
                      bool enable_save,
                      bool enable_metrics,
                      bool infer_only,
                      int input_channels,
                      int input_height,
                      int input_width,
                      int output_channels,
                      int output_height,
                      int output_width);
    ~ZeroDceDataSource() override = default;

    size_t GetTotalFrames() const { return image_paths_.size(); }
    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    std::vector<std::string> image_paths_;
    size_t current_idx_ = 0;
    int input_channels_ = 3;
    int input_height_ = 0;
    int input_width_ = 0;
    int output_channels_ = 3;
    int output_height_ = 0;
    int output_width_ = 0;
    std::filesystem::path input_root_;
    std::filesystem::path gt_root_;
    bool enable_save_ = true;
    bool enable_metrics_ = true;
    bool infer_only_ = false;
    std::unordered_map<std::string, std::string> gt_by_relative_path_;
    std::unordered_map<std::string, std::string> gt_by_filename_;
};
