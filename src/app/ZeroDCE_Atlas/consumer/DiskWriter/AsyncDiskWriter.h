#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>

class AsyncDiskWriter {
public:
    static AsyncDiskWriter& GetInstance();

    void Start(const std::string& output_dir);
    void Stop();
    void Push(uint64_t frame_id, const cv::Mat& img);

private:
    AsyncDiskWriter() = default;
    ~AsyncDiskWriter() { Stop(); }

    struct Task {
        uint64_t frame_id;
        cv::Mat img;
    };

    std::queue<Task> task_queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};
    std::string output_dir_;

    void ProcessQueue();
};