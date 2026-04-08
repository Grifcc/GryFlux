#pragma once

#include "framework/node_base.h"
#include <vector>

class ReidPreprocessNode : public GryFlux::NodeBase {
public:
    /**
     * @param target_w ReID 模型要求的输入宽度 (如 128)
     * @param target_h ReID 模型要求的输入高度 (如 256)
     */
    ReidPreprocessNode(int target_w = 128, int target_h = 256);

    // 执行接口
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int target_w_;
    int target_h_;
};