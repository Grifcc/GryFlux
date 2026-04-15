#pragma once
#include <atomic>
#include <chrono>
#include <iomanip>
#include <string>
#include <tuple>
#include "framework/data_consumer.h"
#include <vector>
#include "../../packet/ZeroDce_Packet.h"

class ZeroDceResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ZeroDceResultConsumer(size_t total_frames);
    ~ZeroDceResultConsumer() = default;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    void printMetrics();

private:
    size_t total_frames_;
    std::atomic<size_t> completed_frames_{0};
    std::vector<std::tuple<std::string, double, double, std::string>> results_log_;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};
