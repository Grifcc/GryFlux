#pragma once

#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    PreprocessNode() = default;
    ~PreprocessNode() override = default;

    // 修复签名，与基类对齐
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};