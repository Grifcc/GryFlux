#pragma once

#include "framework/data_source.h"
#include "packet/deeplab_packet.h"
#include <algorithm>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>

class DeepLabSource : public GryFlux::DataSource {
public:
    // 构造函数：扫描目录，把所有 .jpg 文件的路径存入列表
    DeepLabSource(const std::string& image_dir, const std::string& label_dir = "")
        : label_dir_(label_dir),
          current_idx_(0)
    {
        std::cout << "[Source] 正在扫描数据集目录: " << image_dir << std::endl;
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            if (entry.path().extension() == ".jpg") {
                image_files_.push_back(entry.path().string());
            }
        }
        std::sort(image_files_.begin(), image_files_.end());
        std::cout << "[Source] 扫描完毕，共找到 " << image_files_.size() << " 张图片。" << std::endl;
        setHasMore(!image_files_.empty());
    }

    // 核心函数 2：框架每次调用这个函数，你就吐出一个装好路径的包裹
    std::unique_ptr<GryFlux::DataPacket> produce() override {
        // 如果没数据了，返回 nullptr
        if (current_idx_ >= image_files_.size()) {
            setHasMore(false);
            return nullptr;
        }

        // 1. 创建一个新的、预分配好内存的空白包裹
        auto packet = std::make_unique<DeepLabPacket>();

        // 2. 贴上快递单号 (frame_id) 和 货物地址 (image_path)
        packet->frame_id = current_idx_;
        packet->image_path = image_files_[current_idx_];
        if (!label_dir_.empty()) {
            const std::filesystem::path imagePath(packet->image_path);
            const std::string gtFileName = imagePath.stem().string() + ".png";
            packet->gt_path = (std::filesystem::path(label_dir_) / gtFileName).string();
        }

        // 3. 索引加 1，指向下一张图
        current_idx_++;
        setHasMore(current_idx_ < image_files_.size());

        return packet;
    }

    // 提供一个接口让外部知道总共有多少图（方便传给 Consumer 算进度）
    int getTotalImages() const {
        return static_cast<int>(image_files_.size());
    }

private:
    std::vector<std::string> image_files_;
    std::string label_dir_;
    size_t current_idx_;
};
