#pragma once

#include "framework/node_base.h"

namespace FusionNetV2Nodes
{

class PreprocessNode : public GryFlux::NodeBase
{
public:
    explicit PreprocessNode(int modelWidth = 640, int modelHeight = 480)
        : modelWidth_(modelWidth), modelHeight_(modelHeight) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int modelWidth_ = 640;
    int modelHeight_ = 480;
};

} // namespace FusionNetV2Nodes
