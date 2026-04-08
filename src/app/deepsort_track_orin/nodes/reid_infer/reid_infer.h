#pragma once

#include "framework/node_base.h"
#include <vector>

class ReidInferNode : public GryFlux::NodeBase {
public:
    /**
     * @param feat_dim ReID 模型输出的特征维度 (通常是 512 或 128)
     */
    ReidInferNode(int feat_dim = 512);

    // 核心执行逻辑
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int feat_dim_;
};