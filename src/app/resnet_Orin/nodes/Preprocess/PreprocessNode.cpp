#include "PreprocessNode.h"
#include "../../packet/resnet_packet.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <iostream>
#include <cmath>

constexpr int IMG_WIDTH = 224;
constexpr int IMG_HEIGHT = 224;

const float MEAN_RGB[3] = {0.485f, 0.456f, 0.406f}; 
const float STD_RGB[3]  = {0.229f, 0.224f, 0.225f};

namespace {

struct AxisSample {
    int idx0 = 0;
    int idx1 = 0;
    float w0 = 1.0f;
    float w1 = 0.0f;
};

inline int ClampIndex(int value, int limit) {
    return std::max(0, std::min(value, limit - 1));
}

template <size_t N>
std::array<AxisSample, N> BuildAxisSamples(float scale, int crop_offset, int source_limit) {
    std::array<AxisSample, N> samples;
    for (size_t out_idx = 0; out_idx < N; ++out_idx) {
        const float resized_pos = static_cast<float>(crop_offset + static_cast<int>(out_idx)) + 0.5f;
        const float src_pos = resized_pos * scale - 0.5f;

        const float floor_pos = std::floor(src_pos);
        const int idx0 = ClampIndex(static_cast<int>(floor_pos), source_limit);
        const int idx1 = ClampIndex(idx0 + 1, source_limit);
        const float frac = src_pos - floor_pos;

        samples[out_idx] = AxisSample{
            idx0,
            idx1,
            1.0f - frac,
            frac,
        };
    }
    return samples;
}

}  // namespace

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    cv::Mat image = cv::imread(p.image_path, cv::IMREAD_COLOR);

    if (image.empty()) {
        p.is_valid_image = false;
        p.skipped = true;
        p.skip_reason = "image decode failed";
        std::fill(p.preprocessed_data.begin(), p.preprocessed_data.end(), 0.0f);
        std::cerr << "跳过图片: 无法读取 " << p.image_path << std::endl;
        return;
    }

    int h = image.rows;
    int w = image.cols;
    if (h <= 0 || w <= 0) {
        p.is_valid_image = false;
        p.skipped = true;
        p.skip_reason = "invalid image shape";
        std::fill(p.preprocessed_data.begin(), p.preprocessed_data.end(), 0.0f);
        std::cerr << "跳过图片: 尺寸非法 " << p.image_path << std::endl;
        return;
    }
    int new_w, new_h;
    if (h < w) {
        new_h = 256;
        new_w = std::round(w * (256.0 / h));
    } else {
        new_w = 256;
        new_h = std::round(h * (256.0 / w));
    }

    const float scale_x = static_cast<float>(w) / static_cast<float>(new_w);
    const float scale_y = static_cast<float>(h) / static_cast<float>(new_h);
    const int crop_x = (new_w - IMG_WIDTH) / 2;
    const int crop_y = (new_h - IMG_HEIGHT) / 2;
    const auto x_samples = BuildAxisSamples<IMG_WIDTH>(scale_x, crop_x, w);
    const auto y_samples = BuildAxisSamples<IMG_HEIGHT>(scale_y, crop_y, h);

    p.is_valid_image = true;
    p.skipped = false;
    p.skip_reason.clear();

    int num_pixels = IMG_WIDTH * IMG_HEIGHT;
    float* ptr_r = &p.preprocessed_data[0];
    float* ptr_g = &p.preprocessed_data[num_pixels];
    float* ptr_b = &p.preprocessed_data[num_pixels * 2];
    
    for (int out_y = 0; out_y < IMG_HEIGHT; ++out_y) {
        const AxisSample& ys = y_samples[out_y];
        const unsigned char* row0 = image.ptr<unsigned char>(ys.idx0);
        const unsigned char* row1 = image.ptr<unsigned char>(ys.idx1);

        for (int out_x = 0; out_x < IMG_WIDTH; ++out_x) {
            const AxisSample& xs = x_samples[out_x];
            const unsigned char* p00 = row0 + xs.idx0 * 3;
            const unsigned char* p01 = row0 + xs.idx1 * 3;
            const unsigned char* p10 = row1 + xs.idx0 * 3;
            const unsigned char* p11 = row1 + xs.idx1 * 3;

            const float top_b = xs.w0 * p00[0] + xs.w1 * p01[0];
            const float top_g = xs.w0 * p00[1] + xs.w1 * p01[1];
            const float top_r = xs.w0 * p00[2] + xs.w1 * p01[2];

            const float bottom_b = xs.w0 * p10[0] + xs.w1 * p11[0];
            const float bottom_g = xs.w0 * p10[1] + xs.w1 * p11[1];
            const float bottom_r = xs.w0 * p10[2] + xs.w1 * p11[2];

            const float b = (ys.w0 * top_b + ys.w1 * bottom_b) / 255.0f;
            const float g = (ys.w0 * top_g + ys.w1 * bottom_g) / 255.0f;
            const float r = (ys.w0 * top_r + ys.w1 * bottom_r) / 255.0f;

            *ptr_r++ = (r - MEAN_RGB[0]) / STD_RGB[0];
            *ptr_g++ = (g - MEAN_RGB[1]) / STD_RGB[1];
            *ptr_b++ = (b - MEAN_RGB[2]) / STD_RGB[2];
        }
    }
}
