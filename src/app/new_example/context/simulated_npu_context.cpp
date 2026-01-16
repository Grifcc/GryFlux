/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Simulated NPU Context Implementation
 *************************************************************************************************************************/
#include "simulated_npu_context.h"
#include "utils/logger.h"
#include <algorithm>
#include <stdexcept>

SimulatedNPUContext::SimulatedNPUContext(int deviceId)
    : deviceId_(deviceId), deviceMemory_(kVecSize)
{
    LOG.info("SimulatedNPU %d initialized", deviceId_);
}

SimulatedNPUContext::~SimulatedNPUContext()
{
    LOG.info("SimulatedNPU %d released", deviceId_);
}

void SimulatedNPUContext::copyToDevice(const std::vector<float> &hostData)
{
    // 模拟 Host -> Device 数据传输 (DMA / PCIe)
    if (hostData.size() != deviceMemory_.size())
    {
        throw std::runtime_error("copyToDevice(hostData): hostData size mismatch");
    }
    std::copy(hostData.begin(), hostData.end(), deviceMemory_.begin());

    // 模拟数据传输延迟（真实场景：PCIe Gen3 x16 约 15 GB/s）
    // 对于 256 * 4 bytes = 1 KB 数据，延迟约 0.07 微秒
    // 这里用 1 ms 模拟更保守的情况
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    LOG.debug("NPU %d: Copied %zu elements to device memory",
             deviceId_, deviceMemory_.size());
}

void SimulatedNPUContext::runCompute(float offset)
{
    // 模拟在 NPU 上执行计算（element-wise addition）
    // 真实场景：启动硬件加速器，配置寄存器，等待完成
    for (size_t i = 0; i < deviceMemory_.size(); ++i)
    {
        deviceMemory_[i] += offset;
    }
    
    // 模拟 NPU 计算时间（真实场景：取决于模型复杂度）
    // 这里假设 256 元素的简单操作需要 8ms
    std::this_thread::sleep_for(std::chrono::milliseconds(8));

    LOG.debug("NPU %d: Computed on %zu elements (offset = %.1f)",
             deviceId_, deviceMemory_.size(), offset);
}

std::vector<float> SimulatedNPUContext::copyFromDevice()
{
    std::vector<float> result(deviceMemory_.size());
    copyFromDevice(result);
    return result;
}

void SimulatedNPUContext::copyFromDevice(std::vector<float> &hostOut)
{
    if (hostOut.size() != deviceMemory_.size())
    {
        throw std::runtime_error("copyFromDevice(hostOut): hostOut size mismatch");
    }

    // 模拟 Device -> Host 数据传输
    std::copy(deviceMemory_.begin(), deviceMemory_.end(), hostOut.begin());

    // 模拟数据传输延迟
    std::this_thread::sleep_for(std::chrono::milliseconds(1));

    LOG.debug("NPU %d: Copied %zu elements from device memory",
             deviceId_, hostOut.size());
}

void SimulatedNPUContext::runInference()
{
    // 兼容旧版接口
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
}
