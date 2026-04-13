#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class InputNode : public GryFlux::NodeBase {
public:
    InputNode() = default;
    ~InputNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;
};

}  // namespace PipelineNodes
