#include "ZeroDceResultConsumer.h"
#include "../../packet/ZeroDce_Packet.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <tuple>

// 使用静态变量存储提取出的指标
static std::vector<std::tuple<std::string, double, double, std::string>> s_metrics_log;

ZeroDceResultConsumer::ZeroDceResultConsumer(size_t total_frames) 
    : total_frames_(total_frames), completed_frames_(0) {
    start_time_ = std::chrono::high_resolution_clock::now();
    s_metrics_log.clear();

    if (total_frames_ == 0) {
        finish_signaled_.store(true);
        finish_promise_.set_value();
    }
}

void ZeroDceResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* dce_packet = dynamic_cast<ZeroDcePacket*>(packet.get());
    
    // 抄数据
    s_metrics_log.push_back({
        dce_packet->image_name, 
        dce_packet->int8_psnr, 
        dce_packet->loss, 
        dce_packet->status
    });

    completed_frames_++;
    std::cout << "\r[Consumer] 进度: " << completed_frames_ << " / " << total_frames_ << std::flush;

    // 🌟 终极机制：最后一张图处理完，打表格并解锁主线程
    if (completed_frames_ == total_frames_) {
        printMetrics(); // 打印漂亮的表格

        // 唤醒主线程，允许程序优雅退出
        bool expected = false;
        if (finish_signaled_.compare_exchange_strong(expected, true)) {
            finish_promise_.set_value();
        }
    }
}

// ⬇️ 就是这个函数被你不小心删了，现在给它补回来！
void ZeroDceResultConsumer::printMetrics() {
    // 防御性编程：防止主线程(main)和 consume 重复调用导致打两遍表格
    if (s_metrics_log.empty()) return; 

    std::cout << "\n\n========================================================================\n";
    std::cout << std::left 
              << std::setw(15) << "Image" 
              << "| " << std::setw(15) << "INT8 PSNR" 
              << "| " << std::setw(15) << "Loss" 
              << "| " << "Status\n";
    std::cout << "========================================================================\n";

    double total_psnr = 0.0;
    double total_loss = 0.0;

    for (const auto& log : s_metrics_log) {
        std::cout << std::left 
                  << std::setw(15) << std::get<0>(log) 
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15) << std::get<1>(log) 
                  << "| " << std::fixed << std::setprecision(2) << std::setw(15) << std::get<2>(log) 
                  << "| " << std::get<3>(log) << "\n";
        
        total_psnr += std::get<1>(log);
        total_loss += std::get<2>(log);
    }

    size_t count = completed_frames_ > 0 ? (size_t)completed_frames_ : 1;
    std::cout << "========================================================================\n";
    std::cout << std::left 
              << std::setw(15) << "Average" 
              << "| " << std::fixed << std::setprecision(2) << std::setw(15) << (total_psnr / count) 
              << "| " << std::fixed << std::setprecision(2) << std::setw(15) << (total_loss / count) 
              << "|\n";
    std::cout << "========================================================================\n\n";

    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    double fps = (completed_frames_ * 1000.0) / total_time_ms;

    std::cout << "🚀 流水线部署性能报告 (Pipeline Performance):\n";
    std::cout << "   - 处理总数: " << completed_frames_ << " 张\n";
    std::cout << "   - 总计耗时: " << total_time_ms << " ms\n";
    std::cout << "   - 端到端 FPS: " << std::fixed << std::setprecision(4) << fps << " FPS\n";
    std::cout << "========================================================================\n";

    s_metrics_log.clear(); // 打印完毕后清空，完美闭环
}