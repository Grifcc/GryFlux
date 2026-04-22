#pragma once

#include "framework/node_base.h"

namespace PipelineNodes
{

class PreprocessNode : public GryFlux::NodeBase
{
public:
    explicit PreprocessNode(int delayMs = 0) : delayMs_(delayMs) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int delayMs_;
};

} // namespace PipelineNodes
