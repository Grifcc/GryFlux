#pragma once

#include "framework/node_base.h"
#include "packet/calc_packet.h"

namespace TestNodes
{

class AddConstNode : public GryFlux::NodeBase
{
public:
    using Field = std::vector<double> CalcPacket::*;
    AddConstNode(Field in, Field out, double addend, long long sleepMs = 0)
        : in_(in), out_(out), addend_(addend), sleepMs_(sleepMs)
    {
    }
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    Field in_;
    Field out_;
    double addend_;
    long long sleepMs_;
};

class Add2Node : public GryFlux::NodeBase
{
public:
    using Field = std::vector<double> CalcPacket::*;
    Add2Node(Field in1, Field in2, Field out, long long sleepMs = 0)
        : in1_(in1), in2_(in2), out_(out), sleepMs_(sleepMs)
    {
    }
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    Field in1_;
    Field in2_;
    Field out_;
    long long sleepMs_;
};

class Fuse3SumNode : public GryFlux::NodeBase
{
public:
    using Field = std::vector<double> CalcPacket::*;
    Fuse3SumNode(Field in1, Field in2, Field in3, Field out, long long sleepMs = 0)
        : in1_(in1), in2_(in2), in3_(in3), out_(out), sleepMs_(sleepMs)
    {
    }
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    Field in1_;
    Field in2_;
    Field in3_;
    Field out_;
    long long sleepMs_;
};

} // namespace TestNodes
