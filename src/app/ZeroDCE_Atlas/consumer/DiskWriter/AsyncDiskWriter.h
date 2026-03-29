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
    // 单例模式，全局唯一的写图队列
    static AsyncDiskWriter& GetInstance();

    // 启动后台线程
    void Start(const std::string& output_dir);
    // 停止线程并清空队列
    void Stop();
    // 供 PostprocessNode 调用的接口：把图像丢进队列就立刻返回
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

    // 后台线程一直在跑的死循环
    void ProcessQueue();
};