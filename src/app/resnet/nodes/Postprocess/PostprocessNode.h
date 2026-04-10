#pragma once

#include "framework/node_base.h"

#include <cstddef>
#include <string>
#include <vector>

namespace ResnetNodes
{

class PostprocessNode : public GryFlux::NodeBase
{
public:
    explicit PostprocessNode(std::vector<std::string> classLabels, std::size_t topKCount = 5)
        : classLabels_(std::move(classLabels)), topKCount_(topKCount)
    {
    }

    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    std::vector<std::string> classLabels_;
    std::size_t topKCount_ = 5;
};

} // namespace ResnetNodes

