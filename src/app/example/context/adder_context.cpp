#include "adder_context.h"

#include "utils/logger.h"

#include <chrono>
#include <thread>

AdderContext::AdderContext(int deviceId) : deviceId_(deviceId)
{
    LOG.info("Adder %d initialized", deviceId_);
}

AdderContext::~AdderContext()
{
    LOG.info("Adder %d released", deviceId_);
}

void AdderContext::add(const std::vector<double> &lhs, const std::vector<double> &rhs, std::vector<double> &out)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const size_t n = lhs.size();
    if (rhs.size() != n)
    {
        // Mismatched input sizes: do nothing (caller bug).
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
        out[i] = l + r;
    }
}

void AdderContext::add(const std::vector<double> &lhs, double rhs, std::vector<double> &out)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const size_t n = lhs.size();
    if (out.size() != n)
    {
        out.resize(n);
    }
    for (size_t i = 0; i < n; ++i)
    {
        const double l = lhs[i];
        out[i] = l + rhs;
    }
}
