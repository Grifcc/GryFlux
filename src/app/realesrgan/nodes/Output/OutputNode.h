#pragma once

#include "framework/node_base.h"

namespace RealesrganNodes
{

class OutputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace RealesrganNodes
