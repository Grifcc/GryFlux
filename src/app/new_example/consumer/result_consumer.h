/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Result Consumer (Example)
 *************************************************************************************************************************/
#pragma once

#include "framework/data_consumer.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"
#include <cmath>

/**
 * @brief 结果消费者 - 示例实现
 *
 * 验证并行管道的处理结果是否正确（使用归约验证）。
 *
 * 期望的变换流程：
 * - Input:           rawVec[i] = id (256 个元素)
 * - ImagePreprocess: preprocessedVec[i] = rawVec[i] * 2        (并行分支1, CPU)
 * - ObjectDetection: detectionVec[i] = rawVec[i] + 10          (并行分支2, NPU)
 * - FeatExtractor:   featureVec[i] = preprocessedVec[i] + 5
 * - ObjectTracker:   trackVec[i] = detectionVec[i] + featureVec[i]
 * - Expected:        trackVec[i] = (id + 10) + (id * 2 + 5) = 3 * id + 15
 * - Reduction:       sum(trackVec) = 256 * (3 * id + 15)
 */
class ResultConsumer : public GryFlux::DataConsumer
{
public:
    ResultConsumer() : successCount_(0), failureCount_(0) {}

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override
    {
        auto &p = static_cast<SimpleDataPacket &>(*packet);

        // 新 DAG 验证：
        // a = id
        // b = a * 2
        // c = a * 3
        // d = a + 3
        // e = b * 2
        // f = b * 3
        // g = b + c + d
        // h = e + f + g
        // i = h + c
        // 推导：i = 19 * id + 3
        const float x = static_cast<float>(p.id);
        const float expectedI = 19.0f * x + 3.0f;

        float error = std::abs(p.iValue - expectedI);
        bool correct = error < 0.001f;

        if (correct)
        {
            LOG.info("Packet %d: ✓ PASS (i = %.1f, expected = %.1f, error = %.6f)",
                     p.id, p.iValue, expectedI, error);
            successCount_++;
        }
        else
        {
            LOG.error("Packet %d: ✗ FAIL (i = %.1f, expected = %.1f, error = %.6f)",
                      p.id, p.iValue, expectedI, error);
            failureCount_++;
        }
    }

    // 获取统计信息
    size_t getSuccessCount() const { return successCount_; }
    size_t getFailureCount() const { return failureCount_; }

private:
    size_t successCount_;
    size_t failureCount_;
};
