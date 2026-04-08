#pragma once

#include "framework/node_base.h"
#include <vector>

class PostprocessNode : public GryFlux::NodeBase {
public:
    /**
     * @brief 构造函数
     * @param model_width  模型输入宽度 (默认 640)
     * @param model_height 模型输入高度 (默认 640)
     * @param conf_thresh  目标置信度阈值
     * @param nms_thresh   非极大值抑制(NMS)的 IoU 阈值
     */
    PostprocessNode(int model_width, int model_height, float conf_thresh = 0.3f, float nms_thresh = 0.45f);
    
    // GryFlux 框架执行接口
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int model_width_;
    int model_height_;
    float conf_thresh_;
    float nms_thresh_;

    // YOLOX 的基础参数配置
    const int num_classes_ = 80;
    const std::vector<int> strides_ = {8, 16, 32}; 
};