#pragma once

#include "framework/context.h"

#include <cstddef>
#include <vector>

class MultiplierContext : public GryFlux::Context
{
public:
    explicit MultiplierContext(int deviceId);
    ~MultiplierContext() override;

    int getDeviceId() const { return deviceId_; }

    void mul(const std::vector<double> &lhs, const std::vector<double> &rhs, std::vector<double> &out);
    void mul(const std::vector<double> &lhs, double rhs, std::vector<double> &out);

private:
    int deviceId_;
};
