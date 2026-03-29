#include "AsyncDiskWriter.h"
#include <iostream>

AsyncDiskWriter& AsyncDiskWriter::GetInstance() {
    static AsyncDiskWriter instance;
    return instance;
}

void AsyncDiskWriter::Start(const std::string& output_dir) {
    output_dir_ = output_dir;
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

void AsyncDiskWriter::Push(uint64_t frame_id, const cv::Mat& img) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        // 注意：这里必须 clone，否则流水线复用 Packet 时会把图像内存覆盖掉
        task_queue_.push({frame_id, img.clone()}); 
    }
    cv_.notify_one(); // 唤醒后台线程干活
}

void AsyncDiskWriter::ProcessQueue() {
    while (is_running_) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [this] { return !task_queue_.empty() || !is_running_; });
            
            if (!is_running_ && task_queue_.empty()) break;
            
            task = task_queue_.front();
            task_queue_.pop();
        }

        // 真正的耗时 I/O 操作在这里执行，完全不影响 NPU 推理主干道
        std::string filename = output_dir_ + "/frame_" + std::to_string(task.frame_id) + ".jpg";
        cv::imwrite(filename, task.img);
    }
}