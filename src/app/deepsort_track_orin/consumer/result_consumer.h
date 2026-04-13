#pragma once

#include "framework/data_consumer.h"

#include <opencv2/opencv.hpp>

#include <map>
#include <memory>
#include <string>

#include "../packet/track_data_packet.h"
#include "../utils/deepsort_tracker.h"

class ResultConsumer : public GryFlux::DataConsumer {
public:
    ResultConsumer(const std::string& output_path, double fps, int width, int height);
    ~ResultConsumer() override;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

private:
    std::unique_ptr<DeepSortTracker> tracker_;
    cv::VideoWriter writer_;
    int expected_frame_id_ = 0;
    std::map<int, std::unique_ptr<GryFlux::DataPacket>> reorder_buffer_;

    void ProcessSequentialFrame(TrackDataPacket* packet);
};
