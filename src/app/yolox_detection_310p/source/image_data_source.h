#pragma once

#include "framework/data_source.h"
#include "../packet/detect_data_packet.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <iostream>

class ImageDataSource : public GryFlux::DataSource {
public:
    explicit ImageDataSource(const std::string& video_path) 
        : cap_(video_path), frame_count_(0) 
    {
        if (!cap_.isOpened()) {
            std::cerr << "[ImageDataSource] 错误: 无法打开视频文件: " << video_path << std::endl;
            setHasMore(false); 
        } else {
            std::cout << "[ImageDataSource] 成功打开视频，FPS: " << cap_.get(cv::CAP_PROP_FPS) << std::endl;
            setHasMore(true);  
            readNextFrame();   // 预读第一帧
        }
    }

    ~ImageDataSource() override {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }


    std::unique_ptr<GryFlux::DataPacket> produce() override {
        // 如果已经没有数据了，返回空指针
        if (!hasMore()) {
            return nullptr;
        }

        auto packet = std::make_unique<DetectDataPacket>();
        packet->original_image = next_frame_.clone();
        packet->frame_id = frame_count_++;
        
        readNextFrame();

        return packet;
    }

    double getFps() const { return cap_.get(cv::CAP_PROP_FPS); }
    int getWidth() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)); }
    int getHeight() const { return static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)); }

private:
    cv::VideoCapture cap_;
    cv::Mat next_frame_;    
    int frame_count_;       

    void readNextFrame() {
        cap_ >> next_frame_;
        if (next_frame_.empty()) {
            setHasMore(false);
            std::cout << "[ImageDataSource] 视频读取完毕。共读取 " << frame_count_ << " 帧。" << std::endl;
        }
    }
};