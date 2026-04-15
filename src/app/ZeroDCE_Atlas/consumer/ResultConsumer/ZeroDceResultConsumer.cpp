#include "ZeroDceResultConsumer.h"
#include "../../packet/ZeroDce_Packet.h"
#include <iostream>

ZeroDceResultConsumer::ZeroDceResultConsumer(size_t total_frames) 
    : total_frames_(total_frames), completed_frames_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
    results_log_.clear();
}

void ZeroDceResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* dce_packet = dynamic_cast<ZeroDcePacket*>(packet.get());
    if (dce_packet == nullptr) {
        std::cerr << "[Consumer] 非法数据包，已跳过。" << std::endl;
        return;
    }
    
    results_log_.push_back({
        dce_packet->image_name, 
        dce_packet->int8_psnr, 
        dce_packet->loss, 
        dce_packet->status
    });

    completed_frames_++;
    std::cout << "\r[Consumer] 进度: " << completed_frames_ << " / " << total_frames_ << std::flush;
}

void ZeroDceResultConsumer::printMetrics() {
    if (results_log_.empty()) {
        std::cout << "\n[INFO] 没有可输出的结果。" << std::endl;
        return;
    }

    std::cout << "\n\n========================================================================\n";
    std::cout << std::left 
              << std::setw(15) << "Image" 
              << "| " << std::setw(15) << "INT8 PSNR" 
              << "| " << std::setw(15) << "Loss" 
              << "| " << "Status\n";
    std::cout << "========================================================================\n";

    double total_psnr = 0.0;
    double total_loss = 0.0;

    for (const auto& log : results_log_) {
        std::cout << std::left 
                  << std::setw(15) << std::get<0>(log) 
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15) << std::get<1>(log) 
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15) << std::get<2>(log) 
                  << "| " << std::get<3>(log) << "\n";
        
        total_psnr += std::get<1>(log);
        total_loss += std::get<2>(log);
    }

    size_t count = completed_frames_ > 0 ? static_cast<size_t>(completed_frames_) : 1;
    std::cout << "========================================================================\n";
    std::cout << std::left 
              << std::setw(15) << "Average" 
              << "| " << std::fixed << std::setprecision(2) << std::setw(15) << (total_psnr / count) 
              << "| " << std::fixed << std::setprecision(2) << std::setw(15) << (total_loss / count) 
              << "|\n";
    std::cout << "========================================================================\n\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    double fps = total_time_ms > 0.0 ? (completed_frames_ * 1000.0) / total_time_ms : 0.0;

    std::cout << "[INFO] 处理完成，共 " << completed_frames_ << " 张图片\n";
    std::cout << "   - 处理总数: " << completed_frames_ << " 张\n";
    std::cout << "   - 总计耗时: " << total_time_ms << " ms\n";
    if (total_time_ms > 0.0) {
        std::cout << "   - 端到端 FPS: " << std::fixed << std::setprecision(4) << fps << " FPS\n";
    } else {
        std::cout << "   - 端到端 FPS: 无法计算（耗时为 0）\n";
    }
    std::cout << "========================================================================\n";

    results_log_.clear();
}
