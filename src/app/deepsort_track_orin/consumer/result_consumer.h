#pragma once

#include "framework/data_consumer.h"
#include "../packet/track_data_packet.h"
#include "../utils/deepsort_tracker.h" // 你的算法工具
#include <opencv2/opencv.hpp>
#include <map>
#include <memory>

class ResultConsumer : public GryFlux::DataConsumer {
public:
    /**
     * @param output_path 视频保存路径
     * @param fps 视频帧率
     * @param width 视频宽度
     * @param height 视频高度
     */
    ResultConsumer(const std::string& output_path, double fps, int width, int height);
    ~ResultConsumer() override;

    /**
     * @brief 框架回调接口，每当一个数据包完成所有 Node 计算后会被送入此处
     */
    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override;

private:
    // --- 追踪与保存资源 ---
    std::unique_ptr<DeepSortTracker> tracker_;
    cv::VideoWriter writer_;

    // --- 乱序重排缓冲区 ---
    int expected_frame_id_ = 0; // 下一帧期望拿到的 ID
    
    // 利用 std::map 的自动排序特性，Key 是 frame_id
    std::map<int, std::unique_ptr<GryFlux::DataPacket>> reorder_buffer_;

    // 内部处理函数：执行真正的追踪与可视化
    void processSequentialFrame(TrackDataPacket* p);
};