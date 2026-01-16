/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Object Detection Node Implementation
 *************************************************************************************************************************/
#include "inference_node.h"
#include "packet/simple_data_packet.h"
#include "context/simulated_npu_context.h"
#include "utils/logger.h"

namespace PipelineNodes
{

void ObjectDetectionNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &npu = static_cast<SimulatedNPUContext &>(ctx);

    LOG.debug("Packet %d: ObjectDetection starting on NPU %d (vec size = %zu)",
             p.id, npu.getDeviceId(), p.rawVec.size());

    // ==================== 真实 NPU 操作流程 ====================
    // 步骤 1: 拷贝数据到 NPU 设备内存 (Host -> Device)
    npu.copyToDevice(p.rawVec);

    // 步骤 2: 在 NPU 上执行计算（模拟目标检测）
    // ObjectDetection: detectionVec[i] = rawVec[i] + 10
    npu.runCompute(10.0f);

    // 步骤 3: 拷贝结果回主机内存 (Device -> Host)
    npu.copyFromDevice(p.detectionVec);
    // ===========================================================

    LOG.debug("Packet %d: ObjectDetection completed on NPU %d (result size = %zu)",
             p.id, npu.getDeviceId(), p.detectionVec.size());
}

} // namespace PipelineNodes
