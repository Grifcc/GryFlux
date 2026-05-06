#include "ZeroDceDataSource.h"
#include <iostream>

ZeroDceDataSource::ZeroDceDataSource(const std::string& input_dir) {
    cv::glob(input_dir + "/*.jpg", image_paths_, false);
    std::vector<cv::String> png_paths;
    cv::glob(input_dir + "/*.png", png_paths, false);
    image_paths_.insert(image_paths_.end(), png_paths.begin(), png_paths.end());

    std::cout << "[Source] 发现待处理图片数量: " << image_paths_.size() << std::endl;
}

std::unique_ptr<GryFlux::DataPacket> ZeroDceDataSource::produce() {
    while (current_idx_ < image_paths_.size()) {
        size_t frame_id = current_idx_;
        const auto& image_path = image_paths_[current_idx_];
        cv::Mat input_image = cv::imread(image_path);
        current_idx_++;

        if (input_image.empty()) {
            std::cerr << "[Source] 读取图片失败，跳过: " << image_path << std::endl;
            continue;
        }

        auto packet = std::make_unique<ZeroDcePacket>();
        packet->frame_id = frame_id;
        packet->input_image = std::move(input_image);
        setHasMore(current_idx_ < image_paths_.size());
        return packet;
    }

    setHasMore(false);
    return nullptr;
}
