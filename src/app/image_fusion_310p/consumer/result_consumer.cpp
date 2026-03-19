#include "result_consumer.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

FusionDataConsumer::FusionDataConsumer(const std::string& saveDir) : saveDir_(saveDir) {
    // 确保保存目录存在
    if (!fs::exists(saveDir_)) {
        fs::create_directories(saveDir_);
    }
}

void FusionDataConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    if (!packet) return;

    // 1. 类型转换为具体的 FusionDataPacket
    auto* fusion_packet = static_cast<FusionDataPacket*>(packet.get());

    // 简单校验后处理是否成功输出了结果图
    if (fusion_packet->fused_result.empty()) {
        std::cerr << "[FusionDataConsumer] 警告: " << fusion_packet->filename 
                  << " 的融合结果为空，未能保存。" << std::endl;
        return;
    }

    // 2. 构造保存路径并写入文件
    std::string savePath = saveDir_ + "/" + fusion_packet->filename;
    
    bool success = cv::imwrite(savePath, fusion_packet->fused_result);
    if (success) {
        std::cout << "[FusionDataConsumer] 成功保存: " << savePath << std::endl;
    } else {
        std::cerr << "[FusionDataConsumer] 保存失败: " << savePath << std::endl;
    }

    // 函数结束时，形参 `packet` 作为 unique_ptr 会自动析构。
    // 这时，我们在 Packet 中预分配的所有 cv::Mat 缓存也会随着一同被安全释放，不会导致内存泄漏。
}