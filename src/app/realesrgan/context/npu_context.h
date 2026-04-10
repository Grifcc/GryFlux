#pragma once

#include "framework/context.h"

#include <opencv2/opencv.hpp>
#include <rknn_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class NpuContext : public GryFlux::Context
{
public:
    explicit NpuContext(int deviceId,
                        const std::string &modelPath,
                        int expectedModelWidth,
                        int expectedModelHeight);
    ~NpuContext() override;

    int getDeviceId() const { return deviceId_; }

    void run(const cv::Mat &inputRgbU8, cv::Mat &srTensorF32);

private:
    static float dequantize(int8_t qnt, int zp, float scale);
    static rknn_core_mask toCoreMask(int deviceId);

    cv::Mat makeOutputMatFromNCHW(const float *data, int c, int h, int w) const;
    cv::Mat makeOutputMatFromNHWC(const float *data, int h, int w, int c) const;

    int deviceId_ = 0;
    std::string modelPath_;

    rknn_context ctx_ = 0;
    int inputNum_ = 0;
    int outputNum_ = 0;

    int expectedModelWidth_ = 256;
    int expectedModelHeight_ = 256;

    int inputWidth_ = 0;
    int inputHeight_ = 0;
    int inputChannels_ = 0;

    rknn_tensor_type inputType_ = RKNN_TENSOR_FLOAT32;
    bool inputQuantized_ = false;
    float inputScale_ = 1.0f;
    int inputZeroPoint_ = 0;
    size_t inputElementSize_ = sizeof(float);

    std::vector<uint8_t> modelData_;
    std::vector<rknn_tensor_attr> inputAttrs_;
    std::vector<rknn_tensor_attr> outputAttrs_;
    std::vector<bool> outputQuantized_;
    std::vector<rknn_tensor_mem *> inputMems_;
    std::vector<rknn_tensor_mem *> outputMems_;
    std::vector<std::shared_ptr<float>> outputCache_;
};
