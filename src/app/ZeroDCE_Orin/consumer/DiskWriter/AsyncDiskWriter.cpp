#include "AsyncDiskWriter.h"

#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

AsyncDiskWriter& AsyncDiskWriter::GetInstance() {
    static AsyncDiskWriter instance;
    return instance;
}

void AsyncDiskWriter::Start(const std::string& output_dir) {
    Stop();

    output_dir_ = output_dir;
    fs::create_directories(output_dir_);

    is_running_ = true;
    worker_thread_ = std::thread(&AsyncDiskWriter::ProcessQueue, this);
}

void AsyncDiskWriter::Stop() {
    is_running_ = false;
    cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void AsyncDiskWriter::Push(const std::string& filename, const cv::Mat& img) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        task_queue_.push({filename, img.clone()});
    }
    cv_.notify_one();
}

void AsyncDiskWriter::ProcessQueue() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !task_queue_.empty() || !is_running_; });

            if (!is_running_ && task_queue_.empty()) {
                break;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        const fs::path output_path = fs::path(output_dir_) / task.filename;
        if (!cv::imwrite(output_path.string(), task.img)) {
            std::cerr << "[WARN] 输出图片写入失败: " << output_path << std::endl;
        }
    }
}
