#pragma once

#include "framework/data_consumer.h"
#include "../packet/detect_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

class ResultConsumer : public GryFlux::DataConsumer {
public:
    ResultConsumer(const std::string& output_path, double fps, int width, int height) 
        : writer_(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height)) 
    {
        if (!writer_.isOpened()) {
            std::cerr << "[ResultConsumer] 错误: 无法创建视频写入器: " << output_path << std::endl;
        } else {
            std::cout << "[ResultConsumer] 视频写入器创建成功: " << output_path << std::endl;
        }
    }

    ~ResultConsumer() override {
        if (writer_.isOpened()) {
            writer_.release();
            std::cout << "[ResultConsumer] 视频保存完毕。" << std::endl;
        }
    }

    /**
     * @brief 框架回调接口：消费处理完的数据包
     */
    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override {
        // 1. 获取包含最终结果的数据包
        auto* p = static_cast<DetectDataPacket*>(packet.get());

        // 2. 遍历检测结果，在原图上画框 (这里为了简化，只处理 person=0 和 car=2)
        int valid_count = 0;
        for (const auto& det : p->detections) {
            if (det.class_id == 0 || det.class_id == 2) {
                // 绘制矩形框
                cv::rectangle(p->original_image, 
                              cv::Point(det.x1, det.y1), 
                              cv::Point(det.x2, det.y2), 
                              cv::Scalar(0, 255, 0), 2);
                
                // 绘制置信度和类别
                std::string label = (det.class_id == 0 ? "Person " : "Car ") + std::to_string(det.score).substr(0, 4);
                cv::putText(p->original_image, label, cv::Point(det.x1, det.y1 - 5), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
                
                valid_count++;
            }
        }

        // 3. 写入视频文件
        if (writer_.isOpened()) {
            writer_.write(p->original_image);
        }

        // 4. 打印进度日志 (单线程打印，不会产生错乱)
        std::cout << "[Consumer] 帧 " << p->frame_id << " 处理完毕 | 检测到有效目标数: " << valid_count << std::endl;
        
        // 当函数结束时，packet 的生命周期结束，智能指针会自动释放图像和 NCHW 等内存
    }

private:
    cv::VideoWriter writer_;
};