#pragma once

#include "framework/node_base.h"

namespace PipelineNodes {

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode() = default;
    ~PreprocessNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;
};

}  // namespace PipelineNodes
