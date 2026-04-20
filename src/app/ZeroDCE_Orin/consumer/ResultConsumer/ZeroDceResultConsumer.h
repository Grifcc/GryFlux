#pragma once

#include "../../packet/zero_dce_packet.h"
#include "framework/data_consumer.h"

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <vector>

class ZeroDceResultConsumer : public GryFlux::DataConsumer {
public:
    ZeroDceResultConsumer(size_t total_frames,
                          bool has_ground_truth,
                          bool enable_metrics,
                          bool infer_only);
    ~ZeroDceResultConsumer() override = default;

    std::shared_future<void> get_future() const { return finish_future_; }
    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;
    void printMetrics();
    void signalFailure();

private:
    struct ResultEntry {
        uint64_t frame_id = 0;
        std::string image_name;
        std::string status;
        std::string error_message;
        double int8_psnr = 0.0;
        double loss = 0.0;
        bool write_enqueued = false;
        bool is_proxy_psnr = true;
        bool is_valid_image = true;
        double preprocess_ms = 0.0;
        double infer_ms = 0.0;
        double postprocess_ms = 0.0;
    };

    size_t total_frames_ = 0;
    std::atomic<size_t> completed_frames_{0};
    std::atomic<size_t> skipped_frames_{0};
    std::promise<void> finish_promise_;
    std::shared_future<void> finish_future_;
    std::atomic<bool> finish_signaled_{false};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    bool has_ground_truth_ = false;
    bool enable_metrics_ = true;
    bool infer_only_ = false;

    std::mutex results_mutex_;
    std::vector<ResultEntry> results_;
};
