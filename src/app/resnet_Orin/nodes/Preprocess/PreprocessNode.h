#pragma once
#include "framework/async_pipeline.h"
#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};