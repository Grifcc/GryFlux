#pragma once

#include "framework/data_consumer.h"
#include "packet/calc_packet.h"
#include "utils/logger.h"

#include <cmath>
#include <cstddef>

class CalcConsumer : public GryFlux::DataConsumer
{
public:
    explicit CalcConsumer(size_t printFirstN = 0, bool printAll = false)
        : printFirstN_(printFirstN), printAll_(printAll)
    {
    }

    void consume(std::unique_ptr<GryFlux::DataPacket> packet) override
    {
        auto &p = static_cast<CalcPacket &>(*packet);
        // Expected from test_main.cpp formula:
        // x = id
        // y = x + 1
        // z = x * 2
        // a = y + 10
        // b = y * 3
        // c = y + z
        // d = a + b + c
        // f = d + z = 9 * id + 15
        const double expected = 9.0 * static_cast<double>(p.id) + 15.0;
        double maxErr = 0.0;
        for (const auto &e : p.f)
        {
            const double err = std::abs(e - expected);
            if (err > maxErr)
            {
                maxErr = err;
            }
        }
        const bool ok = maxErr < 1e-9;

        const bool shouldPrint = (!ok) || printAll_ || (printedCount_ < printFirstN_);
        if (shouldPrint)
        {
            const double x0 = p.x.empty() ? 0.0 : p.x[0];
            const double y0 = p.y.empty() ? 0.0 : p.y[0];
            const double z0 = p.z.empty() ? 0.0 : p.z[0];
            const double a0 = p.a.empty() ? 0.0 : p.a[0];
            const double b0 = p.b.empty() ? 0.0 : p.b[0];
            const double c0 = p.c.empty() ? 0.0 : p.c[0];
            const double d0 = p.d.empty() ? 0.0 : p.d[0];
            const double f0 = p.f.empty() ? 0.0 : p.f[0];
            LOG.info(
                "Packet %d: %s | x=%.0f y=%.0f z=%.0f a=%.0f b=%.0f c=%.0f d=%.0f | f=%.6f expected=%.6f maxErr=%.6e vecSize=%zu",
                p.id,
                ok ? "OK" : "FAIL",
                x0,
                y0,
                z0,
                a0,
                b0,
                c0,
                d0,
                f0,
                expected,
                maxErr,
                p.f.size());
            printedCount_++;
        }

        if (ok)
        {
            successCount_++;
        }
        else
        {
            failureCount_++;
            // Always mark error loudly (even if we also printed an info line above)
            const double f0 = p.f.empty() ? 0.0 : p.f[0];
            LOG.error("Packet %d: FAIL f0=%.6f expected=%.6f maxErr=%.6e", p.id, f0, expected, maxErr);
        }
    }

    size_t getSuccessCount() const { return successCount_; }
    size_t getFailureCount() const { return failureCount_; }

private:
    size_t printFirstN_ = 0;
    bool printAll_ = false;
    size_t printedCount_ = 0;
    size_t successCount_ = 0;
    size_t failureCount_ = 0;
};
