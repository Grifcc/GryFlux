#pragma once

#include "framework/node_base.h"

namespace RealesrganNodes
{

class PreprocessNode : public GryFlux::NodeBase
{
public:
    explicit PreprocessNode(int modelWidth = 256, int modelHeight = 256)
        : modelWidth_(modelWidth), modelHeight_(modelHeight) {}

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int modelWidth_ = 256;
    int modelHeight_ = 256;
};

} // namespace RealesrganNodes
