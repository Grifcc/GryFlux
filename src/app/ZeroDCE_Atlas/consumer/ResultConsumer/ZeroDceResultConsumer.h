#pragma once
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "framework/data_consumer.h"
#include "../../packet/ZeroDce_Packet.h"

class ZeroDceResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ZeroDceResultConsumer(size_t total_frames);
    ~ZeroDceResultConsumer() = default;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

    void printMetrics();

private:
    struct ResultItem {
        size_t frame_id;
        std::string image_name;
        double psnr;
        double loss;
        std::string status;
    };

    const size_t total_frames_;
    std::vector<ResultItem> results_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};
