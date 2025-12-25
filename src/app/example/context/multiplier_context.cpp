#include "multiplier_context.h"

#include "utils/logger.h"

#include <chrono>
#include <thread>

MultiplierContext::MultiplierContext(int deviceId) : deviceId_(deviceId)
{
    LOG.info("Multiplier %d initialized", deviceId_);
}

MultiplierContext::~MultiplierContext()
{
    LOG.info("Multiplier %d released", deviceId_);
}

void MultiplierContext::mul(const std::vector<double> &lhs, const std::vector<double> &rhs, std::vector<double> &out)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    const size_t n = lhs.size();
    if (rhs.size() != n)
    {
        return;
    }
    if (out.size() != n)
    {
        out.resize(n);
    }
    for (size_t i = 0; i < n; ++i)
    {
        const double l = lhs[i];
        const double r = rhs[i];
        out[i] = l * r;
    }
}

void MultiplierContext::mul(const std::vector<double> &lhs, double rhs, std::vector<double> &out)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    const size_t n = lhs.size();
    if (out.size() != n)
    {
        out.resize(n);
    }
    for (size_t i = 0; i < n; ++i)
    {
        const double l = lhs[i];
        out[i] = l * rhs;
    }
}
