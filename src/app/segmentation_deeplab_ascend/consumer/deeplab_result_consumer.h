#pragma once

#include "framework/data_consumer.h"

#include <opencv2/core.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

class DeepLabResultConsumer : public GryFlux::DataConsumer
{
public:
    explicit DeepLabResultConsumer(size_t expectedTotal = 0);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    size_t getConsumedCount() const;
    void printSummary() const;

private:
    void updateConfusionMatrix(const cv::Mat &predMask, const cv::Mat &gtMask);

    size_t expectedTotal_;
    std::vector<std::string> classNames_;

    mutable std::mutex mutex_;
    std::vector<int64_t> hist_;
    std::atomic<size_t> consumedCount_{0};
    double totalPacketMiou_{0.0};
};
