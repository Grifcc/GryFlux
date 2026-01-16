/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Preprocess Node
 *************************************************************************************************************************/
#pragma once

#include "framework/node_base.h"
#include "framework/data_packet.h"
#include "framework/context.h"

namespace PipelineNodes
{

/**
 * @brief Image Preprocess Node - 图像预处理（CPU任务）
 *
 * 变换：preprocessedValue = rawValue * 2
 *
 * 这个节点与 ObjectDetectionNode 并行执行！
 */
class ImagePreprocessNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
