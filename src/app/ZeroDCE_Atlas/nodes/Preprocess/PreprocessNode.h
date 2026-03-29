#pragma once
// 这里引入 GryFlux 的基础节点头文件，假设叫 Node.h
// #include "gryflux/Node.h" 
#include "../../packet/ZeroDce_Packet.h"
#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode() = default;
    ~PreprocessNode() = default;

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};