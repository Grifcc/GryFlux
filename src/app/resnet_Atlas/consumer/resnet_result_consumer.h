#pragma once

#include "framework/data_consumer.h"
#include "packet/resnet_packet.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>

class ResNetResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ResNetResultConsumer(size_t total_images);
    ~ResNetResultConsumer() override = default;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    void printMetrics() const;

private:
    size_t total_images_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> top1_correct_{0};
    std::atomic<int> top5_correct_{0};
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
};
