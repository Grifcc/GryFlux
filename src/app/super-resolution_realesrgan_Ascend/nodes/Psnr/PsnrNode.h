#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class PsnrNode : public GryFlux::NodeBase
{
public:
    explicit PsnrNode(int delayMs = 0) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
