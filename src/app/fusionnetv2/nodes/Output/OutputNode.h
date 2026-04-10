#pragma once

#include "framework/node_base.h"

namespace FusionNetV2Nodes
{

class OutputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace FusionNetV2Nodes
