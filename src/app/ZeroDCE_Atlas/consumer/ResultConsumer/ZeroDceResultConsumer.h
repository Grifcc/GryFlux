#pragma once
#include <future>
#include <atomic>
#include <chrono>
#include "framework/data_consumer.h"
#include <vector>
#include "../../packet/ZeroDce_Packet.h"

class ZeroDceResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ZeroDceResultConsumer(size_t total_frames);
    ~ZeroDceResultConsumer() = default;

    std::future<void> get_future() { return finish_promise_.get_future(); }

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    void printMetrics();

private:
    size_t total_frames_;
    std::atomic<size_t> completed_frames_{0};
    std::promise<void> finish_promise_;
    std::atomic<bool> finish_signaled_{false};
    std::vector<ZeroDcePacket*> results_log_;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};