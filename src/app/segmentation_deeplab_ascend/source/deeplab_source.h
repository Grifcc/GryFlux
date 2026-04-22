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
    
    explicit DeepLabSource(const std::string& image_dir)
        : current_idx_(0)
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

    
    std::unique_ptr<GryFlux::DataPacket> produce() override {
        // 如果没数据了，返回 nullptr
        if (current_idx_ >= image_files_.size()) {
            setHasMore(false);
            return nullptr;
        }

        
        auto packet = std::make_unique<DeepLabPacket>();

        
        packet->frame_id = current_idx_;
        packet->image_path = image_files_[current_idx_];

        
        current_idx_++;
        setHasMore(current_idx_ < image_files_.size());

        return packet;
    }

    
    int getTotalImages() const {
        return static_cast<int>(image_files_.size());
    }

private:
    std::vector<std::string> image_files_;
    size_t current_idx_;
};
