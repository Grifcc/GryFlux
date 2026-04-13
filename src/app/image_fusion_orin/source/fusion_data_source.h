#pragma once

#include "framework/data_source.h"

#include <cstdint>
#include <string>
#include <vector>

class FusionDataSource : public GryFlux::DataSource {
public:
    FusionDataSource(
        const std::string& vis_dir,
        const std::string& ir_dir,
        int model_width,
        int model_height);
    ~FusionDataSource() override = default;

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    struct FramePair {
        std::string filename;
        std::string vis_path;
        std::string ir_path;
    };

    static bool IsSupportedImageFile(const std::string& extension);

    std::vector<FramePair> frame_pairs_;
    size_t next_index_ = 0;
    int model_width_ = 0;
    int model_height_ = 0;
};
