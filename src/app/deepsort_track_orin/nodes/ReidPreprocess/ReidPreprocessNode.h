#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class ReidPreprocessNode : public GryFlux::NodeBase {
public:
    ReidPreprocessNode(int target_width, int target_height);
    ~ReidPreprocessNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;

private:
    int target_width_ = 0;
    int target_height_ = 0;
};

}  // namespace PipelineNodes
