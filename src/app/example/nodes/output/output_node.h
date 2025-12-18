/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Output Node
 *************************************************************************************************************************/
#pragma once

#include "framework/node_base.h"
#include "framework/data_packet.h"
#include "framework/context.h"

namespace PipelineNodes
{

/**
 * @brief Object Tracker Node - 目标跟踪（依赖跨帧状态）
 *
 * 变换：trackValue = detectionValue + featureValue
 *
 * 这是一个融合节点，依赖两个前置节点：
 * - ObjectDetectionNode（检测结果）
 * - FeatExtractorNode（特征结果）
 */
class ObjectTrackerNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

/**
 * @brief Final output node - 标记完成（不做数据变换）
 */
class FinalOutputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
