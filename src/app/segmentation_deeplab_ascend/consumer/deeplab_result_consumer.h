#pragma once

#include "framework/data_consumer.h"

#include <atomic>

class DeepLabResultConsumer : public GryFlux::DataConsumer
{
public:
    explicit DeepLabResultConsumer(size_t expectedTotal = 0);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    size_t getConsumedCount() const;
    void printSummary() const;

    size_t expectedTotal_;
    std::atomic<size_t> consumedCount_{0};
};
