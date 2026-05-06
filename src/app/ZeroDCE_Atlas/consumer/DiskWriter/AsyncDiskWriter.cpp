#include "AsyncDiskWriter.h"

#include <filesystem>
#include <iostream>

AsyncDiskWriter& AsyncDiskWriter::GetInstance() {
    static AsyncDiskWriter instance;
    return instance;
}

void AsyncDiskWriter::Start(const std::string& output_dir) {
    std::lock_guard<std::mutex> lock(mtx_);
    output_dir_ = output_dir;

    if (output_dir_.empty()) {
        std::cerr << "[WARN] 输出目录为空，后续图片不会写盘。" << std::endl;
        return;
    }

    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) {
        std::cerr << "[WARN] 创建输出目录失败: " << output_dir_
                  << ", error: " << ec.message() << std::endl;
    }
}

void AsyncDiskWriter::Stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    output_dir_.clear();
}

void AsyncDiskWriter::Push(uint64_t frame_id, const cv::Mat& img) {
    if (img.empty()) {
        std::cerr << "[WARN] 跳过空图像，frame_id=" << frame_id << std::endl;
        return;
    }

    std::string output_dir;
    {
        std::lock_guard<std::mutex> lock(mtx_);
        output_dir = output_dir_;
    }

    if (output_dir.empty()) {
        std::cerr << "[WARN] 输出目录未初始化，跳过写盘，frame_id=" << frame_id << std::endl;
        return;
    }

    const std::string filename = output_dir + "/frame_" + std::to_string(frame_id) + ".jpg";
    if (!cv::imwrite(filename, img)) {
        std::cerr << "[WARN] 写盘失败: " << filename << std::endl;
    }
}
