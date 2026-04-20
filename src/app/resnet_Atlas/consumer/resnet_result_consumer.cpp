#include "consumer/resnet_result_consumer.h"

#include "utils/logger.h"

#include <chrono>
#include <iostream>

ResNetResultConsumer::ResNetResultConsumer(size_t total_images)
    : total_images_(total_images),
      start_time_(std::chrono::steady_clock::now()) {}

void ResNetResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* p = static_cast<ResNetPacket*>(packet.get());

    if (p->top1_class == p->ground_truth_label) {
        ++top1_correct_;
    }
    if (p->top5_correct) {
        ++top5_correct_;
    }

    const int current = ++processed_count_;
    if (current % 100 == 0) {
        LOG.info("[ResNetResultConsumer] Processed %d / %zu images",
                 current,
                 total_images_);
    }
}

void ResNetResultConsumer::printMetrics() const {
    const auto end_time = std::chrono::steady_clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time_);
    const double elapsed_seconds = elapsed_ms.count() / 1000.0;

    std::cout << "\n========================================\n";
    std::cout << "[INFO] Processed " << total_images_ << " images in total\n";
    std::cout << "Total time  : " << elapsed_seconds << " s\n";

    if (elapsed_seconds > 0.0) {
        std::cout << "Throughput  : "
                  << static_cast<double>(total_images_) / elapsed_seconds
                  << " fps\n";
    } else {
        std::cout << "Throughput  : N/A (elapsed time is 0)\n";
    }

    if (total_images_ > 0) {
        std::cout << "Top-1 Acc   : "
                  << static_cast<float>(top1_correct_) / static_cast<float>(total_images_) * 100.0f
                  << "%\n";
        std::cout << "Top-5 Acc   : "
                  << static_cast<float>(top5_correct_) / static_cast<float>(total_images_) * 100.0f
                  << "%\n";
    } else {
        std::cout << "Top-1 Acc   : N/A (total_images_ is 0)\n";
        std::cout << "Top-5 Acc   : N/A (total_images_ is 0)\n";
    }
    std::cout << "========================================\n";
}
