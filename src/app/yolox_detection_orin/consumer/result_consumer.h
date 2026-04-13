#pragma once

#include "framework/data_consumer.h"

#include <opencv2/opencv.hpp>

#include <string>

class ResultConsumer : public GryFlux::DataConsumer {
public:
    ResultConsumer(
        const std::string& output_path,
        double fps,
        int width,
        int height);
    ~ResultConsumer() override;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

private:
    cv::VideoWriter writer_;
};
