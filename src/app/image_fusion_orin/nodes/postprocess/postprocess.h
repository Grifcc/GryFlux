#pragma once

#include "framework/node_base.h"

class PostprocessNode : public GryFlux::NodeBase {
public:
    PostprocessNode() = default;
    ~PostprocessNode() override = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};