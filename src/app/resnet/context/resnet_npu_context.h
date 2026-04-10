#pragma once

#include "framework/context.h"

#include <rknn_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class ResnetNpuContext : public GryFlux::Context
{
public:
    explicit ResnetNpuContext(int deviceId,
                              const std::string &modelPath,
                              int expectedModelWidth,
                              int expectedModelHeight);
    ~ResnetNpuContext() override;

    void run(const std::vector<uint8_t> &inputData,
             std::vector<float> &logits);

private:
    static rknn_core_mask toCoreMask(int deviceId);
    static void resolveInputShape(const rknn_tensor_attr &attr, int &height, int &width, int &channels);

    void releaseResources();

    int deviceId_ = 0;
    std::string modelPath_;
    int expectedModelWidth_ = 224;
    int expectedModelHeight_ = 224;

    rknn_context ctx_ = 0;
    bool initialized_ = false;

    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int inputChannels_ = 0;

    std::vector<std::uint8_t> modelData_;
    rknn_tensor_attr inputAttr_{};
    std::vector<rknn_tensor_attr> outputAttrs_;
};

