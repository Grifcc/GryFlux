#include "postprocess.h"
#include "../../packet/detect_data_packet.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>

namespace {

    inline int clamp(float val, int min, int max) {
        return val > min ? (val < max ? val : max) : min;
    }

    int quick_sort_indice_inverse(std::vector<float> &input, int left, int right, std::vector<int> &indices) {
        float key;
        int key_index;
        int low = left;
        int high = right;
        if (left < right) {
            key_index = indices[left];
            key = input[left];
            while (low < high) {
                while (low < high && input[high] <= key) { high--; }
                input[low] = input[high];
                indices[low] = indices[high];
                while (low < high && input[low] >= key) { low++; }
                input[high] = input[low];
                indices[high] = indices[low];
            }
            input[low] = key;
            indices[low] = key_index;
            quick_sort_indice_inverse(input, left, low - 1, indices);
            quick_sort_indice_inverse(input, low + 1, right, indices);
        }
        return low;
    }

    float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1, float ymax1) {
        float w = std::fmax(0.f, std::fmin(xmax0, xmax1) - std::fmax(xmin0, xmin1) + 1.0f);
        float h = std::fmax(0.f, std::fmin(ymax0, ymax1) - std::fmax(ymin0, ymin1) + 1.0f);
        float i = w * h;
        float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) + (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
        return u <= 0.f ? 0.f : (i / u);
    }

    int nms_filter(int validCount, std::vector<float> &outputLocations, std::vector<int>& classIds, std::vector<int> &order, int filterId, float threshold) {
        for (int i = 0; i < validCount; ++i) {
            int n = order[i];
            if (n == -1 || classIds[n] != filterId) { continue; }

            for (int j = i + 1; j < validCount; ++j) {
                int m = order[j];
                if (m == -1 || classIds[m] != filterId) { continue; }

                float xmin0 = outputLocations[n * 4 + 0];
                float ymin0 = outputLocations[n * 4 + 1];
                float xmax0 = xmin0 + outputLocations[n * 4 + 2];
                float ymax0 = ymin0 + outputLocations[n * 4 + 3];

                float xmin1 = outputLocations[m * 4 + 0];
                float ymin1 = outputLocations[m * 4 + 1];
                float xmax1 = xmin1 + outputLocations[m * 4 + 2];
                float ymax1 = ymin1 + outputLocations[m * 4 + 3];

                float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);

                if (iou > threshold) { order[j] = -1; }
            }
        }
        return 0;
    }

} 

PostprocessNode::PostprocessNode(int model_width, int model_height, float conf_thresh, float nms_thresh)
    : model_width_(model_width), model_height_(model_height), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh) {}

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<DetectDataPacket&>(packet);
    (void)ctx;

    if (p.infer_outputs.size() < 9 || p.infer_outputs[0].empty()) {
        std::cerr << "[PostprocessNode] 错误: 帧 " << p.frame_id << " 收到空的推理数据！" << std::endl;
        return;
    }

    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classIds;
    int validCount = 0;

    for (size_t i = 0; i < strides_.size(); ++i) {
        int stride = strides_[i];
        int grid_h = model_height_ / stride;
        int grid_w = model_width_ / stride;
        
        const float* reg_data = p.infer_outputs[i * 3 + 0].data();
        const float* obj_data = p.infer_outputs[i * 3 + 1].data();
        const float* cls_data = p.infer_outputs[i * 3 + 2].data();

        for (int h = 0; h < grid_h; ++h) {
            for (int w = 0; w < grid_w; ++w) {
                int grid_idx_base = (h * grid_w + w);

                float obj_score = obj_data[grid_idx_base]; 
                obj_score = 1.0f / (1.0f + std::exp(-obj_score)); // Sigmoid
                if (obj_score < conf_thresh_) continue;

                int cls_idx_base = grid_idx_base * num_classes_;
                float max_class_score = 0.0f;
                int max_class_id = -1;
                for (int c = 0; c < num_classes_; ++c) {
                    float class_score = cls_data[cls_idx_base + c];
                    class_score = 1.0f / (1.0f + std::exp(-class_score)); // Sigmoid
                    if (class_score > max_class_score) {
                        max_class_score = class_score;
                        max_class_id = c;
                    }
                }

                float final_score = obj_score * max_class_score;
                if (final_score < conf_thresh_) continue;

                int box_idx_base = grid_idx_base * 4;
                float cx_grid = reg_data[box_idx_base + 0];
                float cy_grid = reg_data[box_idx_base + 1];
                float w_grid  = reg_data[box_idx_base + 2];
                float h_grid  = reg_data[box_idx_base + 3];

                float cx = (cx_grid + w) * stride; 
                float cy = (cy_grid + h) * stride; 
                float box_w = std::exp(w_grid) * stride;
                float box_h = std::exp(h_grid) * stride;
                float box_x = cx - box_w / 2.0f; // 左上角 x
                float box_y = cy - box_h / 2.0f; // 左上角 y

                objProbs.push_back(final_score);
                classIds.push_back(max_class_id);
                filterBoxes.push_back(box_x);
                filterBoxes.push_back(box_y);
                filterBoxes.push_back(box_w);
                filterBoxes.push_back(box_h);
                validCount++;
            }
        }
    }

    if (validCount <= 0) return;

    std::vector<int> indexArray(validCount);
    for (int i = 0; i < validCount; ++i) indexArray[i] = i;

    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(std::begin(classIds), std::end(classIds));
    for (auto c : class_set) {
        nms_filter(validCount, filterBoxes, classIds, indexArray, c, nms_thresh_);
    }

    for (int i = 0; i < validCount; ++i) {
        if (indexArray[i] == -1) continue; 

        int n = indexArray[i]; 
        float x1 = filterBoxes[n * 4 + 0];
        float y1 = filterBoxes[n * 4 + 1];
        float w  = filterBoxes[n * 4 + 2];
        float h  = filterBoxes[n * 4 + 3];
        float x2 = x1 + w;
        float y2 = y1 + h;

        float final_x1 = (clamp(x1, 0, model_width_) - p.preproc_data.x_offset) / p.preproc_data.scale;
        float final_y1 = (clamp(y1, 0, model_height_) - p.preproc_data.y_offset) / p.preproc_data.scale;
        float final_x2 = (clamp(x2, 0, model_width_) - p.preproc_data.x_offset) / p.preproc_data.scale;
        float final_y2 = (clamp(y2, 0, model_height_) - p.preproc_data.y_offset) / p.preproc_data.scale;

        final_x1 = std::max(0.0f, std::min(final_x1, (float)p.preproc_data.original_width));
        final_y1 = std::max(0.0f, std::min(final_y1, (float)p.preproc_data.original_height));
        final_x2 = std::max(0.0f, std::min(final_x2, (float)p.preproc_data.original_width));
        final_y2 = std::max(0.0f, std::min(final_y2, (float)p.preproc_data.original_height));

        Detection det;
        det.x1 = final_x1;
        det.y1 = final_y1;
        det.x2 = final_x2;
        det.y2 = final_y2;
        det.score = objProbs[i];
        det.class_id = classIds[n];
        
        p.detections.push_back(det);
    }
}