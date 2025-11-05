/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Simple Data Packet
 *************************************************************************************************************************/
#pragma once

#include "framework/data_packet.h"
#include <vector>

/**
 * @brief Simple Data Packet for parallel pipeline demonstration
 *
 * 演示并行节点的数据包结构，使用 vector 数据模拟真实场景。
 *
 * DAG 结构：
 *
 *   Input
 *     ├─→ ImagePreprocess ─→ FeatExtractor ─┐
 *     └─→ ObjectDetection(NPU) ──────────────→ ObjectTracker
 *
 * 变换流程（可验证）：
 * - Input:           rawVec[i] = id (填充 256 个元素)
 * - ImagePreprocess: preprocessedVec[i] = rawVec[i] * 2        (并行分支1, CPU)
 * - ObjectDetection: detectionVec[i] = rawVec[i] + 10          (并行分支2, NPU)
 * - FeatExtractor:   featureVec[i] = preprocessedVec[i] + 5
 * - ObjectTracker:   trackVec[i] = detectionVec[i] + featureVec[i]
 * - Consumer:        验证 sum(trackVec) == 256 * (3 * id + 15)
 *
 * 关键设计：
 * - 并行节点（ImagePreprocess 和 ObjectDetection）写入不同字段
 * - 每个节点有独立的输出 vector，避免数据竞争
 * - ObjectDetection 在 NPU 上执行，模拟真实硬件加速
 */
struct SimpleDataPacket : public GryFlux::DataPacket
{
    int id;  // 数据包编号

    // 各节点的输出字段（避免并行节点冲突）
    std::vector<float> rawVec;           // Input 节点设置
    std::vector<float> preprocessedVec;  // ImagePreprocess 输出 (CPU)
    std::vector<float> detectionVec;     // ObjectDetection 输出 (NPU, 并行!)
    std::vector<float> featureVec;       // FeatExtractor 输出 (CPU)
    std::vector<float> trackVec;         // ObjectTracker 输出 (融合结果)

    SimpleDataPacket() : id(0) {}

    uint64_t getId() const override
    {
        return static_cast<uint64_t>(id);
    }
};
