#pragma once

#include "framework/data_source.h"

#include <opencv2/opencv.hpp>

#include <string>

class ImageDataSource : public GryFlux::DataSource {
public:
    ImageDataSource(
        const std::string& video_path,
        int model_width,
        int model_height);
    ~ImageDataSource() override;

    std::unique_ptr<GryFlux::DataPacket> produce() override;

    double getFps() const { return cap_.get(cv::CAP_PROP_FPS); }
    int getWidth() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)); }
    int getHeight() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)); }

private:
    void ReadNextFrame();

    cv::VideoCapture cap_;
    cv::Mat next_frame_;
    int frame_count_ = 0;
    int model_width_ = 0;
    int model_height_ = 0;
};
