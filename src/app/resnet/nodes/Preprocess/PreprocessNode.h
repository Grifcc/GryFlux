#pragma once

#include "framework/node_base.h"

#include <cstddef>

namespace ResnetNodes
{

class PreprocessNode : public GryFlux::NodeBase
{
public:
    explicit PreprocessNode(std::size_t modelWidth = 224, std::size_t modelHeight = 224)
        : modelWidth_(modelWidth), modelHeight_(modelHeight)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::size_t modelWidth_ = 224;
    std::size_t modelHeight_ = 224;
};

} // namespace ResnetNodes

