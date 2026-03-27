#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class InferenceNode : public GryFlux::NodeBase
{
public:
    InferenceNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
