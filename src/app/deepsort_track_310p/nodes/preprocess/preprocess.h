#pragma once

#include "framework/node_base.h"

class PreprocessNode : public GryFlux::NodeBase {
public:
    /**
     * @brief 构造函数，传入模型期望的输入尺寸
     * @param model_width  模型输入宽度 (通常为 640)
     * @param model_height 模型输入高度 (通常为 640)
     */
    PreprocessNode(int model_width, int model_height);
    
    /**
     * @brief GryFlux 框架执行接口
     * @param packet 数据包，流经整个 DAG 的数据载体
     * @param ctx    硬件资源上下文 (本节点为纯 CPU 节点，忽略此参数)
     */
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;

private:
    int model_width_;
    int model_height_;
};