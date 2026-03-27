#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class MiouNode : public GryFlux::NodeBase
{
public:
    explicit MiouNode(int delayMs = 0) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
