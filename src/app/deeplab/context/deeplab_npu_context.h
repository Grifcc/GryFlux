#pragma once

#include "framework/context.h"
#include "packet/deeplab_packet.h"

#include <rknn_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class DeeplabNpuContext : public GryFlux::Context
{
public:
    explicit DeeplabNpuContext(int deviceId,
                               const std::string &modelPath,
                               int expectedModelWidth,
                               int expectedModelHeight);
    ~DeeplabNpuContext() override;

    void run(const std::vector<uint8_t> &inputData,
             std::vector<DeeplabPacket::InferenceOutput> &outputs);

private:
    static rknn_core_mask toCoreMask(int deviceId);
    static void resolveInputShape(const rknn_tensor_attr &attr, int &height, int &width, int &channels);

    void releaseResources();

    int deviceId_ = 0;
    std::string modelPath_;
    int expectedModelWidth_ = 513;
    int expectedModelHeight_ = 513;

    rknn_context ctx_ = 0;
    bool initialized_ = false;

    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int inputChannels_ = 0;
    rknn_tensor_type inputType_ = RKNN_TENSOR_UINT8;
    float inputScale_ = 1.0f;
    int inputZeroPoint_ = 0;

    std::vector<std::uint8_t> modelData_;
    rknn_tensor_attr inputAttr_{};

    std::vector<rknn_tensor_attr> outputAttrs_;
};
