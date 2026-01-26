#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class IAddNode : public GryFlux::NodeBase
{
public:
    IAddNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
