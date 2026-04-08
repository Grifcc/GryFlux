#pragma once

#include "framework/node_base.h"

class InferNode : public GryFlux::NodeBase {
public:
    InferNode() = default;
    ~InferNode() override = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};