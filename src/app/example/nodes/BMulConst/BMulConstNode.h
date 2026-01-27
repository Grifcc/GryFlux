
#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class BMulConstNode : public GryFlux::NodeBase
{
public:
	BMulConstNode() = default;

	void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

};

} // namespace PipelineNodes
