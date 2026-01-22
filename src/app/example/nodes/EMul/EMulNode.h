#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class EMulNode : public GryFlux::NodeBase
{
public:
    explicit EMulNode(int delayMs) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
