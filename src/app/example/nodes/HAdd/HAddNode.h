#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class HAddNode : public GryFlux::NodeBase
{
public:
    HAddNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
