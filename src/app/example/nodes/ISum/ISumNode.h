#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class ISumNode : public GryFlux::NodeBase
{
public:
    explicit ISumNode(int delayMs) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
