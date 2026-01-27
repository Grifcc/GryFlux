#pragma once

#include "framework/context.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

class AdderContext : public GryFlux::Context
{
public:
    explicit AdderContext(int deviceId, int delayMs = 0)
        : deviceId_(deviceId), delayMs_(delayMs) {}

    int getDeviceId() const { return deviceId_; }

    // vector + constant addition
    template <typename T>
    void compute(std::vector<T> &output, const std::vector<T> &input, T value)
    {
        checkSize(output, input);

        const size_t n = input.size();
        for (size_t i = 0; i < n; ++i)
        {
            output[i] = input[i] + value;
        }
        applyDelay();
    }

    // multiple vector addition
    template <typename T, typename... Args>
    void compute(std::vector<T> &output, const std::vector<T> &first, const Args &...rest)
    {
        checkSize(output, first);
        (checkSize(output, rest), ...);

        const size_t n = output.size();
        for (size_t i = 0; i < n; ++i)
        {
            output[i] = first[i] + (rest[i] + ...);
        }
        applyDelay();
    }


private:
    template <typename T>
    static void checkSize(const std::vector<T> &output, const std::vector<T> &input)
    {
        if (output.size() != input.size())
        {
            throw std::runtime_error("Size mismatch: vectors must have the same dimension.");
        }
    }

    void applyDelay()
    {
        if (delayMs_ > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(delayMs_));
        }
    }

    int deviceId_ = 0;
    int delayMs_ = 0;
};