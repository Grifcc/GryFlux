#pragma once
#include "framework/node_base.h"
#include "../../packet/ZeroDce_Packet.h"

class PostprocessNode : public GryFlux::NodeBase {
public:
    PostprocessNode() = default;
    ~PostprocessNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};