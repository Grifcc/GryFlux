#pragma once

#include "framework/data_consumer.h"

#include <atomic>
#include <mutex>

class RealEsrganResultConsumer : public GryFlux::DataConsumer
{
public:
    explicit RealEsrganResultConsumer(size_t expectedTotal = 0);

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    size_t getConsumedCount() const;
    void printSummary() const;

private:
    size_t expectedTotal_;
    mutable std::mutex mutex_;
    std::atomic<size_t> consumedCount_{0};
    size_t validPsnrCount_ = 0;
    double totalPsnr_ = 0.0;
};
