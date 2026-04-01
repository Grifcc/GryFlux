#include "PreprocessNode.h"
#include "../../packet/resnet_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath> 

const int IMG_WIDTH = 224;
const int IMG_HEIGHT = 224;

const float MEAN_RGB[3] = {0.485f, 0.456f, 0.406f}; 
const float STD_RGB[3]  = {0.229f, 0.224f, 0.225f};

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    cv::Mat image = cv::imread(p.image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "[WARN] 图片损坏或路径不存在: " << p.image_path << "，已跳过。" << std::endl;
        std::fill(p.preprocessed_data.begin(), p.preprocessed_data.end(), 0.0f);
        return; 
    }

    int h = image.rows;
    int w = image.cols;
    int new_w, new_h;
    if (h < w) {
        new_h = 256;
        new_w = std::round(w * (256.0 / h));
    } else {
        new_w = 256;
        new_h = std::round(h * (256.0 / w));
    }
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_w, new_h));

    int x = (new_w - IMG_WIDTH) / 2;
    int y = (new_h - IMG_HEIGHT) / 2;
    cv::Rect roi(x, y, IMG_WIDTH, IMG_HEIGHT);
    cv::Mat cropped_image = resized_image(roi).clone();

    int num_pixels = IMG_WIDTH * IMG_HEIGHT;
    float* ptr_r = &p.preprocessed_data[0];
    float* ptr_g = &p.preprocessed_data[num_pixels];
    float* ptr_b = &p.preprocessed_data[num_pixels * 2];
    
    const uint8_t* pixel_ptr = cropped_image.data;

    for (int i = 0; i < num_pixels; ++i) {
        float b = static_cast<float>(pixel_ptr[0]) / 255.0f;
        float g = static_cast<float>(pixel_ptr[1]) / 255.0f;
        float r = static_cast<float>(pixel_ptr[2]) / 255.0f;
        pixel_ptr += 3;
        
        *ptr_r++ = (r - MEAN_RGB[0]) / STD_RGB[0];
        *ptr_g++ = (g - MEAN_RGB[1]) / STD_RGB[1];
        *ptr_b++ = (b - MEAN_RGB[2]) / STD_RGB[2];
    }
}