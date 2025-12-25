#pragma once

#include "framework/context.h"

#include <cstddef>
#include <vector>

class AdderContext : public GryFlux::Context
{
public:
    explicit AdderContext(int deviceId);
    ~AdderContext() override;

    int getDeviceId() const { return deviceId_; }

    void add(const std::vector<double> &lhs, const std::vector<double> &rhs, std::vector<double> &out);
    void add(const std::vector<double> &lhs, double rhs, std::vector<double> &out);

private:
    int deviceId_;
};
