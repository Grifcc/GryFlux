#pragma once
#include "../../packet/ZeroDce_Packet.h"
#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode() = default;
    ~PreprocessNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};