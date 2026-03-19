#pragma once

#include "framework/data_packet.h"
#include <opencv2/opencv.hpp>
#include <vector>

// 确保引用了你 utils 目录下的轨迹定义
#include "../utils/track.h"

/**
 * @brief YOLOX 检测器的中间预处理数据
 */
struct PreprocessData {
    std::vector<float> nchw_data; 
    float scale = 1.0f;
    int x_offset = 0;
    int y_offset = 0;
    int original_width = 0;
    int original_height = 0;
};

/**
 * @brief 单个目标的检测框信息
 */
struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};

/**
 * @brief 追踪任务核心数据包
 * 继承自 GryFlux::DataPacket
 */
struct TrackDataPacket : public GryFlux::DataPacket {
    
    // --- [数据源阶段] ---
    int frame_id = 0;               // 用于消费者重排的唯一帧 ID
    cv::Mat original_image;         // 原始图像，用于 reid_preprocess 抠图

    // --- [YOLOX 检测阶段] ---
    PreprocessData preproc_data;
    std::vector<std::vector<float>> infer_outputs; // 原始特征图输出
    std::vector<Detection> detections;             // NMS 后的检测框

    // --- [ReID 阶段 A: reid_preprocess 写入] ---
    /**
     * @brief 中转仓库：存放所有人/车目标的 NCHW 像素数据
     * 每一行 std::vector<float> 对应 detections 中的一个目标
     * 维度通常为: 3 (通道) * 256 (高) * 128 (宽)
     */
    std::vector<std::vector<float>> reid_preproc_crops;

    // --- [ReID 阶段 B: reid_infer 写入] ---
    /**
     * @brief 最终特征：存放提取出的 ReID 特征向量 (如 512 维)
     */
    std::vector<std::vector<float>> reid_features;

    // --- [追踪阶段: ResultConsumer 写入] ---
    std::vector<Track> active_tracks;
    /**
     * @brief 构造函数，执行内存预分配以提升高并发性能
     */
    TrackDataPacket() {
        infer_outputs.resize(9);
        
        // 预分配容量，避免多线程环境下 vector 频繁扩容导致的性能抖动
        detections.reserve(100); 
        reid_preproc_crops.reserve(100); 
        reid_features.reserve(100); 
        active_tracks.reserve(100);
    }

    /**
     * @brief 实现基类要求的纯虚函数
     * @return 返回帧 ID 作为数据包索引
     */
    uint64_t getIdx() const override {
        return static_cast<uint64_t>(frame_id);
    }
};