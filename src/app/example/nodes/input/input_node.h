#pragma once

#include "framework/node_base.h"

namespace TestNodes
{

class InputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace TestNodes
