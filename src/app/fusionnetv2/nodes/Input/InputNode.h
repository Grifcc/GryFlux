#pragma once

#include "framework/node_base.h"

namespace FusionNetV2Nodes
{

class InputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace FusionNetV2Nodes
