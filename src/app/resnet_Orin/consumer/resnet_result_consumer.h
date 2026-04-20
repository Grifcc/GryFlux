#pragma once

#include "framework/data_consumer.h"
#include "packet/resnet_packet.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <iostream>

class ResNetResultConsumer : public GryFlux::DataConsumer {
public:
    explicit ResNetResultConsumer(size_t total) : total_images_(total) {
        start_time_ = std::chrono::steady_clock::now();
        if (total_images_ == 0) {
            done_signaled_ = true;
            done_promise_.set_value();
        }
    }

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override {
        auto p = static_cast<ResNetPacket*>(packet.get());
        int current = ++completed_count_;

        if (p->skipped || !p->is_valid_image) {
            skipped_count_++;
        } else {
            valid_count_++;
            if (p->top1_class == p->ground_truth_label) top1_correct_++;
            if (p->top5_correct) top5_correct_++;
        }

        if (current % 100 == 0) {
            std::cout << "[INFO] 已处理 " << current << " / " << total_images_ << " 张图片..." << std::endl;
        }

        if (current == total_images_) {
            bool expected = false;
            if (done_signaled_.compare_exchange_strong(expected, true)) {
                done_promise_.set_value();
            }
        }
    }

    std::future<void> getFuture() { return done_promise_.get_future(); }

    void printMetrics() {
        const auto end_time = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = end_time - start_time_;
        const int valid_images = valid_count_.load();
        const int skipped_images = skipped_count_.load();
        const double top1 =
            valid_images > 0 ? static_cast<double>(top1_correct_.load()) / valid_images * 100.0
                             : 0.0;
        const double top5 =
            valid_images > 0 ? static_cast<double>(top5_correct_.load()) / valid_images * 100.0
                             : 0.0;

        std::cout << "\n========================================" << std::endl;
        std::cout << "[INFO] 所有 " << total_images_ << " 张图片已处理完成。" << std::endl;
        std::cout << "总耗时: " << diff.count() << " 秒" << std::endl;
        std::cout << "吞吐量 (FPS): " << total_images_ / diff.count() << " 帧/秒" << std::endl;
        std::cout << "有效图片数: " << valid_images << std::endl;
        std::cout << "跳过图片数: " << skipped_images << std::endl;
        std::cout << "Top-1 准确率: " << top1 << "%" << std::endl;
        std::cout << "Top-5 准确率: " << top5 << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }

private:
    size_t total_images_;
    std::atomic<int> completed_count_{0};
    std::atomic<int> valid_count_{0};
    std::atomic<int> skipped_count_{0};
    std::atomic<int> top1_correct_{0};
    std::atomic<int> top5_correct_{0};
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::promise<void> done_promise_;
    std::atomic<bool> done_signaled_{false};
};
