/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Inference Node
 *************************************************************************************************************************/
#pragma once

#include "framework/node_base.h"
#include "framework/data_packet.h"
#include "framework/context.h"

namespace PipelineNodes
{

/**
 * @brief Object Detection Node - 目标检测（NPU任务）
 *
 * 变换：detectionValue = rawValue + 10
 *
 * 这个节点与 ImagePreprocessNode 并行执行！
 * 使用 NPU 资源进行推理。
 */
class ObjectDetectionNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
