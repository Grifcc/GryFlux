#include "preprocess.h"
#include "../../packet/track_data_packet.h"
#include <iostream>

PreprocessNode::PreprocessNode(int model_width, int model_height)
    : model_width_(model_width), model_height_(model_height) {}

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<TrackDataPacket&>(packet);
    
    (void)ctx; 

    if (p.original_image.empty()) {
        std::cerr << "[PreprocessNode] 警告: 帧 " << p.frame_id << " 收到空图像！" << std::endl;
        return; 
    }

    cv::Mat img;
    cv::cvtColor(p.original_image, img, cv::COLOR_BGR2RGB);

    int img_w = img.cols;
    int img_h = img.rows;
    p.preproc_data.original_width = img_w;
    p.preproc_data.original_height = img_h;

    cv::Mat processed_float_img;

    if (img_w == model_width_ && img_h == model_height_) {
        img.convertTo(processed_float_img, CV_32FC3);
        p.preproc_data.scale = 1.0f;
        p.preproc_data.x_offset = 0;
        p.preproc_data.y_offset = 0;
    } else {
        float scale = std::min(static_cast<float>(model_width_) / img_w,
                               static_cast<float>(model_height_) / img_h);
        p.preproc_data.scale = scale;

        int new_w = static_cast<int>(img_w * scale);
        int new_h = static_cast<int>(img_h * scale);

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(new_w, new_h));

        cv::Mat letterbox_img = cv::Mat(model_height_, model_width_, CV_8UC3, cv::Scalar(114, 114, 114));
        
        p.preproc_data.x_offset = (model_width_ - new_w) / 2;
        p.preproc_data.y_offset = (model_height_ - new_h) / 2;
        
        resized_img.copyTo(letterbox_img(cv::Rect(p.preproc_data.x_offset, p.preproc_data.y_offset, new_w, new_h)));
        
        letterbox_img.convertTo(processed_float_img, CV_32FC3);
    }

    int channels = 3;
    int channel_plane_size = model_height_ * model_width_;
    
    p.preproc_data.nchw_data.resize(channels * channel_plane_size);
    float* nchw_ptr = p.preproc_data.nchw_data.data();

    if (processed_float_img.isContinuous()) {
        const float* hwc_ptr = processed_float_img.ptr<float>();
        for (int i = 0; i < channel_plane_size; ++i) {
            nchw_ptr[i]                                    = hwc_ptr[i * 3 + 0]; 
            nchw_ptr[channel_plane_size + i]               = hwc_ptr[i * 3 + 1];
            nchw_ptr[channel_plane_size * 2 + i]           = hwc_ptr[i * 3 + 2];
        }
    } else {
        for (int h = 0; h < model_height_; ++h) {
            for (int w = 0; w < model_width_; ++w) {
                const float* pixel = processed_float_img.ptr<float>(h, w);
                int base_idx = h * model_width_ + w;
                nchw_ptr[base_idx]                              = pixel[0];
                nchw_ptr[channel_plane_size + base_idx]         = pixel[1];
                nchw_ptr[channel_plane_size * 2 + base_idx]     = pixel[2];
            }
        }
    }
}