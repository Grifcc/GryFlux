#pragma once
#include "framework/data_consumer.h"
#include "packet/resnet_packet.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iostream>

class ResNetResultConsumer : public GryFlux::DataConsumer {
public:
    ResNetResultConsumer(size_t total) : total_images_(total) {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override {
        auto p = static_cast<ResNetPacket*>(packet.get());
        int current = ++processed_count_; 

        if (p->top1_class == p->ground_truth_label) top1_correct_++;
        if (p->top5_correct) top5_correct_++;

        if (current % 100 == 0) {
            std::cout << "[INFO] 已处理 " << current << " / " << total_images_ << " 张图片..." << std::endl;
        }

    }

    void printMetrics() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time_;
        double elapsed_seconds = diff.count();

        std::cout << "\n========================================" << std::endl;
        std::cout << "[INFO] 已处理完成，共 " << total_images_ << " 张图片" << std::endl;
        std::cout << "总耗时: " << elapsed_seconds << " 秒" << std::endl;

        if (elapsed_seconds > 0.0) {
            std::cout << "吞吐量 (FPS): " << static_cast<double>(total_images_) / elapsed_seconds << " 帧/秒" << std::endl;
        } else {
            std::cout << "吞吐量 (FPS): 无法计算（耗时为 0）" << std::endl;
        }

        if (total_images_ > 0) {
            std::cout << "Top-1 准确率: " << static_cast<float>(top1_correct_) / total_images_ * 100.0f << "%" << std::endl;
            std::cout << "Top-5 准确率: " << static_cast<float>(top5_correct_) / total_images_ * 100.0f << "%" << std::endl;
        } else {
            std::cout << "Top-1 准确率: 无法计算（图片总数为 0）" << std::endl;
            std::cout << "Top-5 准确率: 无法计算（图片总数为 0）" << std::endl;
        }

        std::cout << "========================================" << std::endl;
    }

private:
    size_t total_images_;
    std::atomic<int> processed_count_{0};
    std::atomic<int> top1_correct_{0};
    std::atomic<int> top5_correct_{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};
