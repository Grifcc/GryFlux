#pragma once

#include "framework/node_base.h"

namespace DeeplabNodes
{

class OutputNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace DeeplabNodes
