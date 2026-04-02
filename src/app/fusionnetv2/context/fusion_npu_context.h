#pragma once

#include "framework/context.h"

#include <opencv2/core.hpp>
#include <rknn_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class FusionNpuContext : public GryFlux::Context
{
public:
    explicit FusionNpuContext(int deviceId,
                              const std::string &modelPath,
                              int expectedModelWidth,
                              int expectedModelHeight);
    ~FusionNpuContext() override;

    void run(const cv::Mat &visYF32,
             const cv::Mat &infraredF32,
             cv::Mat &fusedYF32);

private:
    using ModelData = std::pair<std::unique_ptr<unsigned char[]>, std::size_t>;

    static rknn_core_mask toCoreMask(int deviceId);
    static float determineInputScaling(const rknn_tensor_attr &attr);
    static std::size_t tensorTypeSize(rknn_tensor_type type);
    static void resolveSpatial(const rknn_tensor_attr &attr, int &height, int &width);

    template<typename T>
    static T clampValue(T value, T minValue, T maxValue)
    {
        return std::max(minValue, std::min(maxValue, value));
    }

    static void dumpTensorAttr(const rknn_tensor_attr &attr);

    ModelData loadModel(const std::string &path) const;
    void prepareTensorAttributes();
    void releaseResources();
    void copyInputData(const cv::Mat &mat, std::size_t index);
    cv::Mat fetchOutputData(std::size_t index);

    int deviceId_ = 0;
    std::string modelPath_;
    int expectedModelWidth_ = 640;
    int expectedModelHeight_ = 480;

    rknn_context ctx_ = 0;
    std::vector<rknn_tensor_attr> inputAttrs_;
    std::vector<rknn_tensor_attr> outputAttrs_;
    std::vector<rknn_tensor_mem *> inputMems_;
    std::vector<rknn_tensor_mem *> outputMems_;
    std::vector<float> inputScaling_;
    bool initialized_ = false;
};
