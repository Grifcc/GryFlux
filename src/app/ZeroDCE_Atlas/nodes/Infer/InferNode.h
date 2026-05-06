#pragma once
#include "framework/node_base.h"
#include "../../packet/ZeroDce_Packet.h"
#include "../../context/AtlasContext.h"

class InferNode :public GryFlux::NodeBase {
public:
    InferNode() = default;
    ~InferNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};