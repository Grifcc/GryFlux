/*************************************************************************************************************************
 * Copyright 2025 GKGgood
 *
 * GryFlux Framework - Simple Data Packet
 *************************************************************************************************************************/
#pragma once

#include "framework/data_packet.h"

#include <vector>
#include <cstdint>

/**
 * @brief Simple Data Packet for new_example DAG
 *
 * 该示例使用“预分配的固定长度 vector”来承载每个节点的输出，避免每帧/每节点反复分配内存，
 * 同时便于用归约(sum)做正确性校验。
 *
 * 变换关系（可验证）：
 * - input:    a = id
 * - mul2:     b = a * 2
 * - mul3:     c = a * 3
 * - add3:     d = a + 3
 * - mul4:     e = b * 2
 * - mul6:     f = b * 3
 * - sum_bcd:  g = b + c + d
 * - sum_efg:  h = e + f + g
 * - sum_hc:   i = h + c
 * - output:   j = i
 */
struct SimpleDataPacket : public GryFlux::DataPacket
{
    static constexpr size_t kVecSize = 256;

    int id;  // 数据包编号

    // DAG 中间值/输出（vector，提前分配空间，避免运行时alloc带来的性能开销）
    std::vector<float> aVec;
    std::vector<float> bVec;
    std::vector<float> cVec;
    std::vector<float> dVec;
    std::vector<float> eVec;
    std::vector<float> fVec;
    std::vector<float> gVec;
    std::vector<float> hVec;
    std::vector<float> iVec;
    std::vector<float> jVec;

    SimpleDataPacket()
        : id(0),
          aVec(kVecSize),
          bVec(kVecSize),
          cVec(kVecSize),
          dVec(kVecSize),
          eVec(kVecSize),
          fVec(kVecSize),
          gVec(kVecSize),
          hVec(kVecSize),
          iVec(kVecSize),
          jVec(kVecSize)
    {
    }

    uint64_t getIdx() const override
    {
        return static_cast<uint64_t>(id);
    }
};
