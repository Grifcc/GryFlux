#include "ZeroDceDataSource.h"
#include <iostream>

ZeroDceDataSource::ZeroDceDataSource(const std::string& input_dir) {
    // 使用 OpenCV 扫描目录下所有 jpg 和 png
    cv::glob(input_dir + "/*.jpg", image_paths_, false);
    std::vector<cv::String> png_paths;
    cv::glob(input_dir + "/*.png", png_paths, false);
    image_paths_.insert(image_paths_.end(), png_paths.begin(), png_paths.end());

    std::cout << "[Source] 发现待处理图片数量: " << image_paths_.size() << std::endl;
}

std::unique_ptr<GryFlux::DataPacket> ZeroDceDataSource::produce() {
    if (current_idx_ >= image_paths_.size()) {
        setHasMore(false);
        return nullptr;
    }

    // 注意这里变成了 make_unique
    auto packet = std::make_unique<ZeroDcePacket>(); 

    packet->frame_id = current_idx_;
    packet->input_image = cv::imread(image_paths_[current_idx_]);
    current_idx_++;
    setHasMore(current_idx_ < image_paths_.size());
    return packet; // C++ 会自动向上转型为 DataPacket
}