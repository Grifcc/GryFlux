#pragma once
#include "framework/async_pipeline.h"
#include "framework/node_base.h"

// 🚨 重点检查这里：必须是 PostprocessNode，不能是 PreprocessNode 或 PostProcessNode(大写P)
class PostprocessNode : public GryFlux::NodeBase {
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};