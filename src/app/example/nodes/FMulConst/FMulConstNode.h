#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class FMulConstNode : public GryFlux::NodeBase
{
public:
    FMulConstNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
