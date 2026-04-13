#pragma once

#include "framework/data_packet.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../utils/track.h"

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

struct TrackDataPacket : public GryFlux::DataPacket {
    static constexpr size_t kDefaultOutputSlotCount = 9;
    static constexpr size_t kDefaultDetectionCapacity = 100;
    static constexpr int kNumClasses = 80;

    int frame_id = 0;
    cv::Mat original_image;

    PreprocessData preproc_data;
    std::vector<std::vector<float>> infer_outputs;
    std::vector<Detection> detections;
    std::vector<std::vector<float>> reid_preproc_crops;
    std::vector<uint8_t> reid_crop_valid_flags;
    size_t active_reid_crop_count = 0;
    std::vector<std::vector<float>> reid_features;
    size_t active_reid_feature_count = 0;
    std::vector<Track> active_tracks;
    size_t max_detection_capacity = 0;

    explicit TrackDataPacket(
        int detection_model_width,
        int detection_model_height,
        int reid_width,
        int reid_height,
        int reid_feature_dim,
        size_t output_slot_count = kDefaultOutputSlotCount,
        size_t detection_capacity = kDefaultDetectionCapacity) {
        max_detection_capacity = detection_capacity;
        infer_outputs = BuildDefaultInferOutputs(
            detection_model_width,
            detection_model_height,
            output_slot_count);
        detections.reserve(detection_capacity);
        reid_preproc_crops.resize(detection_capacity);
        reid_crop_valid_flags.resize(detection_capacity, 0U);
        reid_features.resize(detection_capacity);
        active_tracks.reserve(detection_capacity);
        preproc_data.nchw_data.resize(
            static_cast<size_t>(3) * static_cast<size_t>(detection_model_width) *
            static_cast<size_t>(detection_model_height));

        const size_t reid_crop_element_count =
            static_cast<size_t>(3) * static_cast<size_t>(reid_width) *
            static_cast<size_t>(reid_height);
        const size_t reid_feature_element_count =
            static_cast<size_t>(std::max(reid_feature_dim, 0));
        for (size_t index = 0; index < detection_capacity; ++index) {
            reid_preproc_crops[index].resize(reid_crop_element_count);
            reid_features[index].resize(reid_feature_element_count);
        }
    }

    uint64_t getIdx() const override { return static_cast<uint64_t>(frame_id); }

    void ResetFrameState() {
        detections.clear();
        active_tracks.clear();
        active_reid_crop_count = 0;
        active_reid_feature_count = 0;
        std::fill(reid_crop_valid_flags.begin(), reid_crop_valid_flags.end(), 0U);
    }

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
