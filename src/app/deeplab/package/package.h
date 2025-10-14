/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"


class ImagePackage : public GryFlux::DataObject
{
public:

    ImagePackage(const cv::Mat& frame, int idx, float scale = 1.0f, int x_pad = 0, int y_pad = 0)
        :src_frame_(frame), idx_(idx), scale_(scale), x_pad_(x_pad), y_pad_(y_pad) {};
    ~ImagePackage() {};

    const cv::Mat get_data() const {
      return src_frame_;
    }
    int get_id() const {
      return idx_;
    }
    int get_width() const {
      return src_frame_.cols;
    }
    int get_height() const {
      return src_frame_.rows;
    }
    float get_scale() const {
      return scale_;
    }
    int get_x_pad() const {
      return x_pad_;
    }
    int get_y_pad() const {
      return y_pad_;
    }
private:
    cv::Mat src_frame_;
    int idx_;
    float scale_;
    int x_pad_;
    int y_pad_;
};

class RunnerPackage : public GryFlux::DataObject
{
public:
    RunnerPackage(std::size_t model_width, std::size_t model_height):model_width_(model_width), model_height_(model_height) {};
    ~RunnerPackage() {};

    using OutputData = std::pair<std::shared_ptr<float[]>, std::size_t>; //buffer, element count
    using GridSize = std::pair<std::size_t, std::size_t>; // grid height, width

    std::vector<OutputData> get_output() const {
      return rknn_output_buff;
    }
    std::vector<GridSize> get_grid() const {
      return grid_sizes_;
    }
    std::size_t get_model_width() const {
      return model_width_;
    }
    std::size_t get_model_height() const {
      return model_height_;
    }
    void push_data(OutputData output_data, GridSize grid_size) {
      rknn_output_buff.push_back(output_data);
      grid_sizes_.push_back(grid_size);
    }

    std::size_t size() const {
      assert(rknn_output_buff.size() == grid_sizes_.size());
      return rknn_output_buff.size();
    }
private:

    std::size_t model_width_;
    std::size_t model_height_;

    std::vector<OutputData> rknn_output_buff;
    std::vector<GridSize> grid_sizes_;

};


class MaskPackage : public GryFlux::DataObject
{
public:
    MaskPackage(int img_id, const cv::Mat& mask)
        :img_id_(img_id), mask_(mask) {};
    ~MaskPackage() {};

    // mask: CV_8UC1，每个像素值为类别id
    const cv::Mat& get_mask() const {
        return mask_;
    }
    int get_img_id() const {
        return img_id_;
    }
private:
    int img_id_;
    cv::Mat mask_; // 分割mask
};