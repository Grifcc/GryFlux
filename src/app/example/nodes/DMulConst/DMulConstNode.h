#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class DMulConstNode : public GryFlux::NodeBase
{
public:
    explicit DMulConstNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
