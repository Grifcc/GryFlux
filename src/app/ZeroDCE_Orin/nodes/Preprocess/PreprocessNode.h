#pragma once

#include "../../packet/zero_dce_packet.h"
#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode() = default;
    ~PreprocessNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;
};
