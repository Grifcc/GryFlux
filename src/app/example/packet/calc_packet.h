#pragma once

#include "framework/data_packet.h"

#include <cstddef>
#include <vector>

struct CalcPacket : public GryFlux::DataPacket
{
    static constexpr size_t kDefaultVectorSize = 128;

    int id = 0;

    // Layer outputs (write-once fields to avoid data races)
    std::vector<double> x; // input
    std::vector<double> y; // add1
    std::vector<double> z; // mul1
    std::vector<double> a; // add2
    std::vector<double> b; // mul2
    std::vector<double> c; // add3
    std::vector<double> d; // d = a + b + c
    std::vector<double> f; // fuse

    CalcPacket() { init(kDefaultVectorSize); }
    explicit CalcPacket(size_t vectorSize) { init(vectorSize); }

    void init(size_t vectorSize)
    {
        x.resize(vectorSize);
        y.resize(vectorSize);
        z.resize(vectorSize);
        a.resize(vectorSize);
        b.resize(vectorSize);
        c.resize(vectorSize);
        d.resize(vectorSize);
        f.resize(vectorSize);
    }

    uint64_t getIdx() const override { return static_cast<uint64_t>(id); }
};
