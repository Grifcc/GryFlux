#include "ZeroDceResultConsumer.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <utility>

ZeroDceResultConsumer::ZeroDceResultConsumer(size_t total_frames,
                                             bool has_ground_truth,
                                             bool enable_metrics,
                                             bool infer_only)
    : total_frames_(total_frames),
      finish_future_(finish_promise_.get_future().share()),
      has_ground_truth_(has_ground_truth),
      enable_metrics_(enable_metrics),
      infer_only_(infer_only) {
    start_time_ = std::chrono::high_resolution_clock::now();
    results_.reserve(total_frames_);

    if (total_frames_ == 0) {
        finish_signaled_.store(true);
        finish_promise_.set_value();
    }
}

void ZeroDceResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* dce_packet = static_cast<ZeroDcePacket*>(packet.get());

    if (!dce_packet->is_valid_image) {
        skipped_frames_.fetch_add(1);
    }

    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_.push_back(ResultEntry{
            dce_packet->frame_id,
            dce_packet->image_name,
            dce_packet->status,
            dce_packet->error_message,
            dce_packet->int8_psnr,
            dce_packet->loss,
            dce_packet->write_enqueued,
            dce_packet->is_proxy_psnr,
            dce_packet->is_valid_image,
            dce_packet->preprocess_ms,
            dce_packet->infer_ms,
            dce_packet->postprocess_ms,
        });
    }

    const size_t completed = completed_frames_.fetch_add(1) + 1;
    std::cout << "\r[Consumer] 进度: " << completed << " / " << total_frames_ << std::flush;

    if (completed == total_frames_) {
        bool expected = false;
        if (finish_signaled_.compare_exchange_strong(expected, true)) {
            finish_promise_.set_value();
        }
    }
}

void ZeroDceResultConsumer::printMetrics() {
    const auto end_time = std::chrono::high_resolution_clock::now();
    const double total_time_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    const double fps =
        total_time_ms > 0.0 ? (completed_frames_.load() * 1000.0) / total_time_ms : 0.0;

    std::vector<ResultEntry> results_copy;
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        results_copy = results_;
    }

    if (results_copy.empty()) {
        return;
    }

    std::sort(results_copy.begin(), results_copy.end(), [](const ResultEntry& lhs, const ResultEntry& rhs) {
        return lhs.frame_id < rhs.frame_id;
    });

    double total_psnr = 0.0;
    double total_loss = 0.0;
    double total_preprocess_ms = 0.0;
    double total_infer_ms = 0.0;
    double total_postprocess_ms = 0.0;
    size_t valid_count = 0;
    const char* psnr_title = has_ground_truth_ ? "PSNR" : "Proxy PSNR";

    if (enable_metrics_) {
        std::cout << "\n\n========================================================================\n";
        std::cout << std::left
                  << std::setw(15) << "Image"
                  << "| " << std::setw(15) << psnr_title
                  << "| " << std::setw(15) << "Loss"
                  << "| " << "Status\n";
        std::cout << "========================================================================\n";
    }

    for (const auto& result : results_copy) {
        if (enable_metrics_) {
            std::cout << std::left
                      << std::setw(15) << result.image_name
                      << "| " << std::fixed << std::setprecision(2) << std::setw(15) << result.int8_psnr
                      << "| " << std::fixed << std::setprecision(2) << std::setw(15) << result.loss
                      << "| " << result.status << "\n";
        }

        if (result.is_valid_image) {
            total_psnr += result.int8_psnr;
            total_loss += result.loss;
            total_preprocess_ms += result.preprocess_ms;
            total_infer_ms += result.infer_ms;
            total_postprocess_ms += result.postprocess_ms;
            ++valid_count;
        }
    }

    if (enable_metrics_) {
        std::cout << "========================================================================\n";
        std::cout << std::left
                  << std::setw(15) << "Average"
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15)
                  << (valid_count > 0 ? total_psnr / valid_count : 0.0)
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15)
                  << (valid_count > 0 ? total_loss / valid_count : 0.0)
                  << "|\n";
        std::cout << "========================================================================\n\n";
    }

    if (enable_metrics_ && valid_count > 0) {
        std::cout << "Quality Summary:\n";
        std::cout << "  - 平均" << psnr_title << ": "
                  << std::fixed << std::setprecision(2) << (total_psnr / valid_count) << " dB\n";
        std::cout << "  - 平均 Loss: "
                  << std::fixed << std::setprecision(2) << (total_loss / valid_count) << "\n";
    }

    if (valid_count > 0) {
        std::cout << "Stage Latency:\n";
        std::cout << "  - Preprocess 平均耗时: "
                  << std::fixed << std::setprecision(3) << (total_preprocess_ms / valid_count) << " ms\n";
        std::cout << "  - Infer 平均耗时: "
                  << std::fixed << std::setprecision(3) << (total_infer_ms / valid_count) << " ms\n";
        std::cout << "  - Postprocess 平均耗时: "
                  << std::fixed << std::setprecision(3) << (total_postprocess_ms / valid_count) << " ms\n";
    }

    std::cout << "Pipeline Performance:\n";
    if (!enable_metrics_) {
        std::cout << "  - 运行模式: " << (infer_only_ ? "infer-only" : "no-save/no-metrics") << "\n";
    }
    std::cout << "  - 处理总数: " << completed_frames_.load() << " 张\n";
    std::cout << "  - 跳过数量: " << skipped_frames_.load() << " 张\n";
    std::cout << "  - 总计耗时: " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "  - 端到端 FPS: " << std::fixed << std::setprecision(4) << fps << " FPS\n";
    std::cout << "========================================================================\n";
}

void ZeroDceResultConsumer::signalFailure() {
    bool expected = false;
    if (finish_signaled_.compare_exchange_strong(expected, true)) {
        finish_promise_.set_value();
    }
}
