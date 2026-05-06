#pragma once
#include <opencv2/opencv.hpp>
#include <mutex>
#include <string>

class AsyncDiskWriter {
public:
    static AsyncDiskWriter& GetInstance();

    void Start(const std::string& output_dir);
    void Stop();
    void Push(uint64_t frame_id, const cv::Mat& img);

private:
    AsyncDiskWriter() = default;
    ~AsyncDiskWriter() { Stop(); }

    std::mutex mtx_;
    std::string output_dir_;
};
