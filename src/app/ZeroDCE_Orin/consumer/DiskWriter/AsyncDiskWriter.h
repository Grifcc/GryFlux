#pragma once

#include <opencv2/opencv.hpp>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class AsyncDiskWriter {
public:
    static AsyncDiskWriter& GetInstance();

    void Start(const std::string& output_dir);
    void Stop();
    void Push(const std::string& filename, const cv::Mat& img);

private:
    AsyncDiskWriter() = default;
    ~AsyncDiskWriter() { Stop(); }

    struct Task {
        std::string filename;
        cv::Mat img;
    };

    void ProcessQueue();

    std::queue<Task> task_queue_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};
    std::string output_dir_;
};
