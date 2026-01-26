#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class EDivConstNode : public GryFlux::NodeBase
{
public:
    explicit EDivConstNode(int delayMs) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
