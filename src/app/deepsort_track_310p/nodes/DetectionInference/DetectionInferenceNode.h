#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class DetectionInferenceNode : public GryFlux::NodeBase {
public:
    DetectionInferenceNode() = default;
    ~DetectionInferenceNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;
};

}  // namespace PipelineNodes
