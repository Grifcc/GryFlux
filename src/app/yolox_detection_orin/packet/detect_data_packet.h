#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

struct PreprocessData {
    std::vector<float> nchw_data;

    float scale = 1.0f;
    int x_offset = 0;
    int y_offset = 0;
    int original_width = 0;
    int original_height = 0;
};

struct Detection {
    float x1, y1, x2, y2, score;
    int class_id;
};

struct DetectDataPacket : public GryFlux::DataPacket {
    static constexpr size_t kDefaultOutputSlotCount = 9;
    static constexpr size_t kDefaultDetectionCapacity = 100;
    static constexpr int kNumClasses = 80;

    int frame_id = 0;
    cv::Mat original_image;

    PreprocessData preproc_data;
    std::vector<std::vector<float>> infer_outputs;
    std::vector<Detection> detections;

    explicit DetectDataPacket(
        int model_width,
        int model_height,
        size_t output_slot_count = kDefaultOutputSlotCount,
        size_t detection_capacity = kDefaultDetectionCapacity) {
        infer_outputs = BuildDefaultInferOutputs(
            model_width,
            model_height,
            output_slot_count);
        detections.reserve(detection_capacity);
        preproc_data.nchw_data.resize(
            static_cast<size_t>(3) * static_cast<size_t>(model_width) *
            static_cast<size_t>(model_height));
    }

    uint64_t getIdx() const override { return static_cast<uint64_t>(frame_id); }

private:
    static std::vector<std::vector<float>> BuildDefaultInferOutputs(
        int model_width,
        int model_height,
        size_t output_slot_count) {
        std::vector<std::vector<float>> outputs(output_slot_count);
        if (output_slot_count != kDefaultOutputSlotCount) {
            return outputs;
        }

        const int strides[] = {8, 16, 32};
        size_t output_index = 0;
        for (const int stride : strides) {
            const size_t grid_width =
                static_cast<size_t>(std::max(1, model_width / stride));
            const size_t grid_height =
                static_cast<size_t>(std::max(1, model_height / stride));
            const size_t cell_count = grid_width * grid_height;

            outputs[output_index++].resize(cell_count * 4U);
            outputs[output_index++].resize(cell_count);
            outputs[output_index++].resize(cell_count * kNumClasses);
        }

        return outputs;
    }
};
