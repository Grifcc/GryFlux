
#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class BMulNode : public GryFlux::NodeBase
{
public:
	explicit BMulNode(int delayMs) : delayMs_(delayMs) {}

	void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
	int delayMs_;
};

} // namespace PipelineNodes
