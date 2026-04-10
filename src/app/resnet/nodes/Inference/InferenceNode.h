#pragma once

#include "framework/node_base.h"

namespace ResnetNodes
{

class InferenceNode : public GryFlux::NodeBase
{
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace ResnetNodes

