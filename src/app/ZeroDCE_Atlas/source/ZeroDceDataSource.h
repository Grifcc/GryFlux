#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "../packet/ZeroDce_Packet.h"
#include "framework/data_source.h"

// #include "framework/data_source.h" // 根据 GryFlux 的实际基础头文件路径修改

class ZeroDceDataSource : public GryFlux::DataSource {
public:
    explicit ZeroDceDataSource(const std::string& input_dir);
    ~ZeroDceDataSource() = default;

    // 获取总帧数，供 Consumer 和 Main 使用
    size_t GetTotalFrames() const { return image_paths_.size(); }

    // 框架不断调用的拉取数据的接口 (函数名根据 GryFlux 的虚函数修改，比如 GetNext 或者 Fetch)
    //std::shared_ptr<ZeroDcePacket> GetNext();
    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    std::vector<cv::String> image_paths_;
    size_t current_idx_ = 0;
};