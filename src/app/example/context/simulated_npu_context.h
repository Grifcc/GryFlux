/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Simulated NPU Context
 *************************************************************************************************************************/
#pragma once

#include "framework/context.h"
#include <vector>
#include <chrono>
#include <thread>

/**
 * @brief Simulated NPU Context
 *
 * 模拟真实 NPU 设备的操作流程：
 * 1. copyToDevice() - 拷贝数据到 NPU 内存
 * 2. runCompute() - 在 NPU 上执行计算
 * 3. copyFromDevice() - 拷贝结果回主机内存
 *
 * 在真实场景中，这些操作会涉及 DMA 传输、硬件加速器等。
 */
class SimulatedNPUContext : public GryFlux::Context
{
private:
    static constexpr size_t kVecSize = 256;

    int deviceId_;
    std::vector<float> deviceMemory_;  // 模拟 NPU 设备内存

public:
    explicit SimulatedNPUContext(int deviceId);
    ~SimulatedNPUContext();

    int getDeviceId() const { return deviceId_; }

    /**
     * @brief 模拟数据拷贝到 NPU 设备内存 (Host -> Device)
     *
     * 在真实场景中，这会触发 DMA 传输或 PCIe 数据传输。
     * 这里用内存拷贝 + sleep 模拟延迟。
     */
    void copyToDevice(const std::vector<float> &hostData);

    /**
     * @brief 模拟在 NPU 上执行计算
     *
     * 在真实场景中，这会启动硬件加速器执行模型推理。
     * 这里用简单的算术操作 + sleep 模拟。
     *
     * @param offset 对每个元素加的偏移量
     */
    void runCompute(float offset);

    /**
     * @brief 模拟数据从 NPU 拷贝回主机内存 (Device -> Host)
     *
     * 在真实场景中，这会触发反向 DMA 传输。
     */
    std::vector<float> copyFromDevice();

    /**
     * @brief 模拟数据从 NPU 拷贝回主机内存 (Device -> Host)，写入预分配缓冲区
     *
     * @note hostOut 必须提前 resize 到正确大小（示例中由 DataPacket 构造函数预分配）。
     */
    void copyFromDevice(std::vector<float> &hostOut);

    /**
     * @brief 兼容旧版接口（已废弃，推荐使用三步法）
     */
    void runInference();
};
