#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace {

int Clamp(float value, int min_value, int max_value) {
    return value > min_value ? (value < max_value ? value : max_value)
                             : min_value;
}

int QuickSortIndiceInverse(
    std::vector<float>& values,
    int left,
    int right,
    std::vector<int>& indices) {
    float key = 0.0f;
    int key_index = 0;
    int low = left;
    int high = right;
    if (left < right) {
        key_index = indices[left];
        key = values[left];
        while (low < high) {
            while (low < high && values[high] <= key) {
                --high;
            }
            values[low] = values[high];
            indices[low] = indices[high];
            while (low < high && values[low] >= key) {
                ++low;
            }
            values[high] = values[low];
            indices[high] = indices[low];
        }
        values[low] = key;
        indices[low] = key_index;
        QuickSortIndiceInverse(values, left, low - 1, indices);
        QuickSortIndiceInverse(values, low + 1, right, indices);
    }
    return low;
}

float CalculateOverlap(
    float xmin0,
    float ymin0,
    float xmax0,
    float ymax0,
    float xmin1,
    float ymin1,
    float xmax1,
    float ymax1) {
    const float width =
        std::fmax(0.0f, std::fmin(xmax0, xmax1) - std::fmax(xmin0, xmin1) + 1.0f);
    const float height =
        std::fmax(0.0f, std::fmin(ymax0, ymax1) - std::fmax(ymin0, ymin1) + 1.0f);
    const float intersection = width * height;
    const float union_area =
        (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
        (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - intersection;
    return union_area <= 0.0f ? 0.0f : (intersection / union_area);
}

void ApplyNmsFilter(
    int valid_count,
    std::vector<float>& box_locations,
    std::vector<int>& class_ids,
    std::vector<int>& order,
    int filter_id,
    float threshold) {
    for (int first = 0; first < valid_count; ++first) {
        const int selected_index = order[first];
        if (selected_index == -1 || class_ids[selected_index] != filter_id) {
            continue;
        }

        for (int second = first + 1; second < valid_count; ++second) {
            const int candidate_index = order[second];
            if (candidate_index == -1 || class_ids[candidate_index] != filter_id) {
                continue;
            }

            const float xmin0 = box_locations[selected_index * 4 + 0];
            const float ymin0 = box_locations[selected_index * 4 + 1];
            const float xmax0 = xmin0 + box_locations[selected_index * 4 + 2];
            const float ymax0 = ymin0 + box_locations[selected_index * 4 + 3];

            const float xmin1 = box_locations[candidate_index * 4 + 0];
            const float ymin1 = box_locations[candidate_index * 4 + 1];
            const float xmax1 = xmin1 + box_locations[candidate_index * 4 + 2];
            const float ymax1 = ymin1 + box_locations[candidate_index * 4 + 3];

            if (CalculateOverlap(xmin0,
                                 ymin0,
                                 xmax0,
                                 ymax0,
                                 xmin1,
                                 ymin1,
                                 xmax1,
                                 ymax1) > threshold) {
                order[second] = -1;
            }
        }
    }
}

}  // namespace

namespace PipelineNodes {

PostprocessNode::PostprocessNode(
    int model_width,
    int model_height,
    float confidence_threshold,
    float nms_threshold)
    : model_width_(model_width),
      model_height_(model_height),
      confidence_threshold_(confidence_threshold),
      nms_threshold_(nms_threshold) {}

void PostprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& detect_packet = static_cast<DetectDataPacket&>(packet);
    (void)ctx;

    detect_packet.detections.clear();
    if (detect_packet.infer_outputs.size() < 9 || detect_packet.infer_outputs[0].empty()) {
        LOG.error("[PostprocessNode] Packet %d received empty inference output",
                  detect_packet.frame_id);
        detect_packet.markFailed();
        return;
    }

    const size_t total_cell_count =
        static_cast<size_t>(model_width_ / 8) * static_cast<size_t>(model_height_ / 8) +
        static_cast<size_t>(model_width_ / 16) * static_cast<size_t>(model_height_ / 16) +
        static_cast<size_t>(model_width_ / 32) * static_cast<size_t>(model_height_ / 32);

    std::vector<float> filter_boxes;
    std::vector<float> object_probabilities;
    std::vector<int> class_ids;
    filter_boxes.reserve(total_cell_count * 4U);
    object_probabilities.reserve(total_cell_count);
    class_ids.reserve(total_cell_count);

    int valid_count = 0;
    for (size_t stride_index = 0; stride_index < strides_.size(); ++stride_index) {
        const int stride = strides_[stride_index];
        const int grid_height = model_height_ / stride;
        const int grid_width = model_width_ / stride;

        const float* reg_data = detect_packet.infer_outputs[stride_index * 3 + 0].data();
        const float* obj_data = detect_packet.infer_outputs[stride_index * 3 + 1].data();
        const float* cls_data = detect_packet.infer_outputs[stride_index * 3 + 2].data();

        for (int grid_y = 0; grid_y < grid_height; ++grid_y) {
            for (int grid_x = 0; grid_x < grid_width; ++grid_x) {
                const int grid_index = grid_y * grid_width + grid_x;

                float object_score = obj_data[grid_index];
                object_score = 1.0f / (1.0f + std::exp(-object_score));
                if (object_score < confidence_threshold_) {
                    continue;
                }

                const int class_index = grid_index * num_classes_;
                float max_class_score = 0.0f;
                int max_class_id = -1;
                for (int class_id = 0; class_id < num_classes_; ++class_id) {
                    float class_score = cls_data[class_index + class_id];
                    class_score = 1.0f / (1.0f + std::exp(-class_score));
                    if (class_score > max_class_score) {
                        max_class_score = class_score;
                        max_class_id = class_id;
                    }
                }

                const float final_score = object_score * max_class_score;
                if (final_score < confidence_threshold_) {
                    continue;
                }

                const int box_index = grid_index * 4;
                const float center_x =
                    (reg_data[box_index + 0] + static_cast<float>(grid_x)) *
                    static_cast<float>(stride);
                const float center_y =
                    (reg_data[box_index + 1] + static_cast<float>(grid_y)) *
                    static_cast<float>(stride);
                const float width =
                    std::exp(reg_data[box_index + 2]) * static_cast<float>(stride);
                const float height =
                    std::exp(reg_data[box_index + 3]) * static_cast<float>(stride);

                filter_boxes.push_back(center_x - width / 2.0f);
                filter_boxes.push_back(center_y - height / 2.0f);
                filter_boxes.push_back(width);
                filter_boxes.push_back(height);
                object_probabilities.push_back(final_score);
                class_ids.push_back(max_class_id);
                ++valid_count;
            }
        }
    }

    if (valid_count <= 0) {
        return;
    }

    std::vector<int> sorted_indices(valid_count);
    for (int index = 0; index < valid_count; ++index) {
        sorted_indices[index] = index;
    }

    QuickSortIndiceInverse(
        object_probabilities,
        0,
        valid_count - 1,
        sorted_indices);

    const std::set<int> unique_classes(class_ids.begin(), class_ids.end());
    for (const int class_id : unique_classes) {
        ApplyNmsFilter(valid_count,
                       filter_boxes,
                       class_ids,
                       sorted_indices,
                       class_id,
                       nms_threshold_);
    }

    for (int index = 0; index < valid_count; ++index) {
        const int box_id = sorted_indices[index];
        if (box_id == -1) {
            continue;
        }

        const float x1 = filter_boxes[box_id * 4 + 0];
        const float y1 = filter_boxes[box_id * 4 + 1];
        const float x2 = x1 + filter_boxes[box_id * 4 + 2];
        const float y2 = y1 + filter_boxes[box_id * 4 + 3];

        Detection detection;
        detection.x1 = static_cast<float>(
            std::max(0.0,
                     std::min(
                         static_cast<double>((Clamp(x1, 0, model_width_) -
                                              detect_packet.preproc_data.x_offset) /
                                             detect_packet.preproc_data.scale),
                         static_cast<double>(detect_packet.preproc_data.original_width))));
        detection.y1 = static_cast<float>(
            std::max(0.0,
                     std::min(
                         static_cast<double>((Clamp(y1, 0, model_height_) -
                                              detect_packet.preproc_data.y_offset) /
                                             detect_packet.preproc_data.scale),
                         static_cast<double>(detect_packet.preproc_data.original_height))));
        detection.x2 = static_cast<float>(
            std::max(0.0,
                     std::min(
                         static_cast<double>((Clamp(x2, 0, model_width_) -
                                              detect_packet.preproc_data.x_offset) /
                                             detect_packet.preproc_data.scale),
                         static_cast<double>(detect_packet.preproc_data.original_width))));
        detection.y2 = static_cast<float>(
            std::max(0.0,
                     std::min(
                         static_cast<double>((Clamp(y2, 0, model_height_) -
                                              detect_packet.preproc_data.y_offset) /
                                             detect_packet.preproc_data.scale),
                         static_cast<double>(detect_packet.preproc_data.original_height))));
        detection.score = object_probabilities[index];
        detection.class_id = class_ids[box_id];
        detect_packet.detections.push_back(detection);
    }
}

}  // namespace PipelineNodes
