#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class ReidInferenceNode : public GryFlux::NodeBase {
public:
    explicit ReidInferenceNode(int feature_dimension);
    ~ReidInferenceNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;

private:
    int feature_dimension_ = 0;
};

}  // namespace PipelineNodes
