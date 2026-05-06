#include "ZeroDceResultConsumer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>

ZeroDceResultConsumer::ZeroDceResultConsumer(size_t total_frames)
    : total_frames_(total_frames),
      start_time_(std::chrono::high_resolution_clock::now()) {
    results_.reserve(total_frames_);
}

void ZeroDceResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* dce_packet = dynamic_cast<ZeroDcePacket*>(packet.get());
    if (!dce_packet) {
        std::cerr << "[Consumer] 非法数据包，已跳过。" << std::endl;
        return;
    }

    results_.push_back({
        dce_packet->frame_id,
        dce_packet->image_name,
        dce_packet->int8_psnr,
        dce_packet->loss,
        dce_packet->status
    });

    std::cout << "\r[Consumer] 进度: " << results_.size() << " / " << total_frames_ << std::flush;
}

void ZeroDceResultConsumer::printMetrics() {
    if (results_.empty()) {
        std::cout << "\n[INFO] 没有可输出的结果。" << std::endl;
        return;
    }

    double total_psnr = 0.0;
    double total_loss = 0.0;
    std::sort(results_.begin(), results_.end(),
              [](const ResultItem& lhs, const ResultItem& rhs) {
                  return lhs.frame_id < rhs.frame_id;
              });
    std::cout << '\n';
    std::cout << std::fixed << std::setprecision(2);
    const std::string separator =
        "==============================================================";

    std::cout << separator << '\n';
    std::cout << std::left
              << std::setw(14) << "Image"
              << "| " << std::setw(11) << "INT8 PSNR"
              << "| " << std::setw(9) << "Loss"
              << "| " << std::setw(8) << "Status" << '\n';
    std::cout << separator << '\n';

    for (const auto& result : results_) {
        std::cout << std::left
                  << std::setw(14) << result.image_name
                  << "| " << std::setw(11) << result.psnr
                  << "| " << std::setw(9) << result.loss
                  << "| " << std::setw(8) << result.status << '\n';
        total_psnr += result.psnr;
        total_loss += result.loss;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    const double total_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    const size_t count = results_.size();
    const double avg_psnr = total_psnr / static_cast<double>(count);
    const double avg_loss = total_loss / static_cast<double>(count);

    std::cout << separator << '\n';
    std::cout << std::left
              << std::setw(14) << "Average"
              << "| " << std::setw(11) << avg_psnr
              << "| " << std::setw(9) << avg_loss
              << "| " << std::setw(8) << "" << '\n';
    std::cout << separator << "\n\n";

    std::cout << "[INFO] 处理完成，共 " << count << " 张图片\n";
    std::cout << "  - 处理总数: " << count << " 张\n";
    std::cout << "  - 总计耗时: " << total_time_ms << " ms\n";
    if (total_time_ms > 0.0) {
        const double fps = (count * 1000.0) / total_time_ms;
        std::cout << "  - 端到端 FPS: " << std::setprecision(4) << fps << " FPS\n";
        std::cout << std::setprecision(2);
    }
    std::cout << separator << '\n';

    results_.clear();
}
