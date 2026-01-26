#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class GAddNode : public GryFlux::NodeBase
{
public:
    GAddNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
