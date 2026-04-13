#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class OutputNode : public GryFlux::NodeBase {
public:
    OutputNode() = default;
    ~OutputNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;
};

}  // namespace PipelineNodes
