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
    ResNetResultConsumer(size_t total) : total_images_(total) {
        start_time_ = std::chrono::high_resolution_clock::now();
        if (total_images_ == 0) {
            done_signaled_ = true;
            done_promise_.set_value();
        }
    }

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override {
        auto p = static_cast<ResNetPacket*>(packet.get());
        int current = ++processed_count_; 

        if (p->top1_class == p->ground_truth_label) top1_correct_++;
        if (p->top5_correct) top5_correct_++;

        if (current % 100 == 0) {
            std::cout << "[INFO] 已处理 " << current << " / " << total_images_ << " 张图片..." << std::endl;
        }

        if (current == total_images_) {
            bool expected = false;
            if (done_signaled_.compare_exchange_strong(expected, true)) {
                done_promise_.set_value(); // 🌟 关键！处理完最后一张，发送信号！不再 exit(0)
            }
        }
    }

    // 🌟 对外暴露获取信号的接口
    std::future<void> get_future() { return done_promise_.get_future(); }

    void printMetrics() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time_;
        std::cout << "\n========================================" << std::endl;
        std::cout << "[INFO] 🚀 恭喜！所有 " << total_images_ << " 张图片已全部处理完毕！" << std::endl;
        std::cout << "总耗时: " << diff.count() << " 秒" << std::endl;
        std::cout << "吞吐量 (FPS): " << total_images_ / diff.count() << " 帧/秒" << std::endl;
        std::cout << "Top-1 准确率: " << (float)top1_correct_ / total_images_ * 100.0f << "%" << std::endl;
        std::cout << "Top-5 准确率: " << (float)top5_correct_ / total_images_ * 100.0f << "%" << std::endl;
        std::cout << "========================================" << std::endl;
    }

private:
    size_t total_images_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> top1_correct_{0};
    std::atomic<int> top5_correct_{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
    std::promise<void> done_promise_; // 🌟 新增的承诺器
    std::atomic<bool> done_signaled_{false};
};
