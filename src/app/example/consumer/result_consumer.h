/*************************************************************************************************************************
 * Copyright 2025 GKGgood
 *
 * GryFlux Framework - Result Consumer (Example)
 *************************************************************************************************************************/
#pragma once

#include "framework/data_consumer.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

#include <atomic>
#include <cmath>
#include <numeric>

/**
 * @brief 结果消费者 - 示例实现
 *
 * 验证 example 这个 DAG 的输出是否正确。
 *
 * 变换关系：
 * - input:     a = id
 * - BMul:      b = a * 2
 * - CSub:      c = a - 1
 * - DMul:      d = a * 4
 * - EDiv:      e = b / 2
 * - FMul:      f = b * 3
 * - GSum:      g = b + c + d
 * - HSum:      h = e + f + g
 * - ISum:      i = h + c
 * - output:    j = i
 *
 * expect output: j = 18 * id - 1
 */
class ResultConsumer : public GryFlux::DataConsumer
{
public:
    ResultConsumer() = default;

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override
    {
        consumedCount_.fetch_add(1, std::memory_order_relaxed);

        auto &p = static_cast<SimpleDataPacket &>(*packet);

        const float x = static_cast<float>(p.id);
        const float expectedOut = 18.0f * x - 1.0f;

        const float sum = std::accumulate(p.jVec.begin(), p.jVec.end(), 0.0f);
        const float expectedSum = static_cast<float>(SimpleDataPacket::kVecSize) * expectedOut;

        float error = std::abs(sum - expectedSum);
        bool correct = error < 0.001f;

        if (correct)
        {
            LOG.info("Packet %d: ✓ PASS (output[0] = %.1f, sum = %.1f, expectedSum = %.1f, error = %.6f)",
                     p.id, p.jVec.empty() ? 0.0f : p.jVec[0], sum, expectedSum, error);
        }
        else
        {
            LOG.error("Packet %d: ✗ FAIL (output[0] = %.1f, sum = %.1f, expectedSum = %.1f, error = %.6f)",
                     p.id, p.jVec.empty() ? 0.0f : p.jVec[0], sum, expectedSum, error);
        }
    }

    // Statistics
    size_t getConsumedCount() const { return consumedCount_.load(std::memory_order_relaxed); }

private:
    std::atomic<size_t> consumedCount_{0};
    
};
