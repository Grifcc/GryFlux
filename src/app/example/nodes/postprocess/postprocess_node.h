/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Feature Extractor Node
 *************************************************************************************************************************/
#pragma once

#include "framework/node_base.h"
#include "framework/data_packet.h"
#include "framework/context.h"

namespace PipelineNodes
{

/**
 * @brief Feature Extractor Node - 特征提取（CPU任务）
 *
 * 变换：featureValue = preprocessedValue + 5
 *
 * 依赖 ImagePreprocessNode 的输出。
 */
class FeatExtractorNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
