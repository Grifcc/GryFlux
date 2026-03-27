/*************************************************************************************************************************
 * Copyright 2025 FallenSoul-He
 *
 * GryFlux Framework - Simple Data Packet
 *************************************************************************************************************************/
#pragma once

#include "framework/data_packet.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// --- 模型常量定义 ---
constexpr int MODEL_INPUT_W = 513;
constexpr int MODEL_INPUT_H = 513;
constexpr int MODEL_OUT_W = 65;
constexpr int MODEL_OUT_H = 65;
constexpr int NUM_CLASSES = 21;

/**
 * @brief DeepLabV3 流水线专属数据包
 */
struct DeepLabPacket : public GryFlux::DataPacket {
    // ==========================================
    // 1. 元数据
    // ==========================================
    int frame_id = 0;
    std::string image_path;
    std::string gt_path;

    int orig_w = 0;
    int orig_h = 0;
    // ==========================================
    // 2. 预分配的内存坑位 (实现零拷贝的核心)
    // ==========================================
    // 前处理输出 / NPU 输入 (3 * 513 * 513)
    std::vector<float> input_tensor;

    // NPU 输出 (65 * 65 * 21)
    std::vector<float> output_tensor;

    // 后处理输出 (恢复原图大小的预测结果)
    cv::Mat pred_mask_resized;

    // GtProcess 节点输出 (读取并转换好的真值标签)
    cv::Mat gt_mask;

    float miou = 0.f;
    
    // ==========================================
    // 3. 构造函数 (提前开辟内存)
    // ==========================================
    DeepLabPacket()
        : input_tensor(3 * MODEL_INPUT_H * MODEL_INPUT_W),
          output_tensor(MODEL_OUT_H * MODEL_OUT_W * NUM_CLASSES)
    {
    }

    uint64_t getIdx() const override {
        return static_cast<uint64_t>(frame_id);
    }
};