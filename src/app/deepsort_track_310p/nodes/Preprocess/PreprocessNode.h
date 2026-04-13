#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode(int model_width, int model_height);
    ~PreprocessNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;

private:
    int model_width_ = 0;
    int model_height_ = 0;
};

}  // namespace PipelineNodes
