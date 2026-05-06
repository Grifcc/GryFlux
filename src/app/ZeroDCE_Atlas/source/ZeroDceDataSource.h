#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../packet/ZeroDce_Packet.h"
#include "framework/data_source.h"


class ZeroDceDataSource : public GryFlux::DataSource {
public:
    explicit ZeroDceDataSource(const std::string& input_dir);
    ~ZeroDceDataSource() = default;


    size_t GetTotalFrames() const { return image_paths_.size(); }

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    std::vector<cv::String> image_paths_;
    size_t current_idx_ = 0;
};