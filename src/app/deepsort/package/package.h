#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "framework/data_object.h"
#include "opencv2/opencv.hpp"

class ImagePackage : public GryFlux::DataObject
{
public:
    ImagePackage(const cv::Mat &frame, int idx, float scale = 1.0f, int x_pad = 0, int y_pad = 0)
        : src_frame_(frame), idx_(idx), scale_(scale), x_pad_(x_pad), y_pad_(y_pad) {}
    ~ImagePackage() = default;

    const cv::Mat &get_data() const { return src_frame_; }
    int get_id() const { return idx_; }
    int get_width() const { return src_frame_.cols; }
    int get_height() const { return src_frame_.rows; }
    float get_scale() const { return scale_; }
    int get_x_pad() const { return x_pad_; }
    int get_y_pad() const { return y_pad_; }

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
    using OutputData = std::pair<std::shared_ptr<float[]>, std::size_t>;
    using GridSize = std::pair<std::size_t, std::size_t>;

    RunnerPackage(std::size_t model_width, std::size_t model_height)
        : model_width_(model_width), model_height_(model_height) {}
    ~RunnerPackage() = default;

    std::vector<OutputData> get_output() const { return rknn_output_buff_; }
    std::vector<GridSize> get_grid() const { return grid_sizes_; }
    std::size_t get_model_width() const { return model_width_; }
    std::size_t get_model_height() const { return model_height_; }

    void push_data(OutputData output_data, GridSize grid_size)
    {
        rknn_output_buff_.push_back(output_data);
        grid_sizes_.push_back(grid_size);
    }

    std::size_t size() const
    {
        assert(rknn_output_buff_.size() == grid_sizes_.size());
        return rknn_output_buff_.size();
    }

private:
    std::size_t model_width_;
    std::size_t model_height_;
    std::vector<OutputData> rknn_output_buff_;
    std::vector<GridSize> grid_sizes_;
};

struct ObjectInfo
{
    int left;
    int top;
    int right;
    int bottom;
    int class_id;
    float prob;
};

class ObjectPackage : public GryFlux::DataObject
{
public:
    explicit ObjectPackage(int img_id) : img_id_(img_id) {}
    ~ObjectPackage() = default;

    std::vector<ObjectInfo> get_data() const { return objects_; }
    int get_image_id() const { return img_id_; }
    void push_data(const ObjectInfo &obj_info) { objects_.push_back(obj_info); }

private:
    int img_id_;
    std::vector<ObjectInfo> objects_;
};

struct TrackedObject
{
    int left;
    int top;
    int right;
    int bottom;
    int class_id;
    float prob;
    int track_id;
};

class TrackPackage : public GryFlux::DataObject
{
public:
    explicit TrackPackage(int img_id) : img_id_(img_id) {}
    ~TrackPackage() = default;

    std::vector<TrackedObject> get_data() const { return tracked_objects_; }
    int get_image_id() const { return img_id_; }
    void push_data(const TrackedObject &obj) { tracked_objects_.push_back(obj); }

private:
    int img_id_;
    std::vector<TrackedObject> tracked_objects_;
};
