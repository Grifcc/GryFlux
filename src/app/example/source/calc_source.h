#pragma once

#include "framework/data_source.h"
#include "packet/calc_packet.h"

class CalcSource : public GryFlux::DataSource
{
public:
    explicit CalcSource(size_t totalPackets) : totalPackets_(totalPackets), producedCount_(0)
    {
        setHasMore(totalPackets_ > 0);
    }

    std::unique_ptr<GryFlux::DataPacket> produce() override
    {
        if (producedCount_ >= totalPackets_)
        {
            setHasMore(false);
            return nullptr;
        }

        auto packet = std::make_unique<CalcPacket>();
        packet->id = static_cast<int>(producedCount_);
        producedCount_++;
        setHasMore(producedCount_ < totalPackets_);
        return packet;
    }

private:
    size_t totalPackets_;
    size_t producedCount_;
};
