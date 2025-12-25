#pragma once

#include "framework/node_base.h"
#include "packet/calc_packet.h"

namespace TestNodes
{

class MulConstNode : public GryFlux::NodeBase
{
public:
    using Field = std::vector<double> CalcPacket::*;
    MulConstNode(Field in, Field out, double factor, long long sleepMs = 0)
        : in_(in), out_(out), factor_(factor), sleepMs_(sleepMs)
    {
    }
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    Field in_;
    Field out_;
    double factor_;
    long long sleepMs_;
};

} // namespace TestNodes
