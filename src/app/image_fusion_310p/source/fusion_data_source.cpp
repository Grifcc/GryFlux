#include "fusion_data_source.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>

namespace fs = std::filesystem;

FusionDataSource::FusionDataSource(const std::string& visDir, const std::string& irDir)
    : irDir_(irDir) {
    // 检查目录是否存在并初始化迭代器
    if (fs::exists(visDir)) {
        dirIter_ = fs::directory_iterator(visDir);
    } else {
        std::cerr << "[FusionDataSource] 错误: Visible 目录不存在: " << visDir << std::endl;
        dirIter_ = endIter_; 
        setHasMore(false); // 直接停止生产
    }
}

std::unique_ptr<GryFlux::DataPacket> FusionDataSource::produce() {
    // 使用 while 循环是为了跳过读取失败的图片，直到找到一张合法的或者遍历完目录
    while (dirIter_ != endIter_) {
        // 1. 构造文件路径
        std::string visFilename = dirIter_->path().filename().string();
        std::string visPath = dirIter_->path().string();
        std::string irPath = irDir_ + "/" + visFilename; // 假设文件名相同
        
        // 迭代器步进，为下一次 produce 做准备
        ++dirIter_;

        // 2. 读取图像 (Visible 读彩色，IR 读灰度)
        cv::Mat visRaw = cv::imread(visPath, cv::IMREAD_COLOR);
        cv::Mat irRaw = cv::imread(irPath, cv::IMREAD_GRAYSCALE);

        if (visRaw.empty() || irRaw.empty()) {
            std::cerr << "[FusionDataSource] 警告: 图像读取失败或缺失配对，跳过: " << visFilename << std::endl;
            continue; 
        }

        // 3. 创建 Packet 并装载数据
        auto packet = std::make_unique<FusionDataPacket>();
        packet->packet_idx = current_idx_++; // 赋值并自增
        packet->filename = visFilename;
        
        // 【性能优化】：使用 std::move 转移 cv::Mat 内部指针所有权，避免数据深拷贝
        packet->vis_raw = std::move(visRaw); 
        packet->ir_raw = std::move(irRaw);

        // 如果刚好是最后一张图，通知框架后续没有数据了
        if (dirIter_ == endIter_) {
            setHasMore(false);
        }

        return packet;
    }

    // 目录遍历结束
    setHasMore(false);
    return nullptr;
}