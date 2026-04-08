#pragma once

#include "framework/node_base.h"

class InferNode : public GryFlux::NodeBase {
public:
    InferNode() = default;
    ~InferNode() override = default;

    /**
     * @brief GryFlux 框架执行接口
     * @param packet 数据包 (包含预处理好的 NCHW 数据，以及用于存放输出的容器)
     * @param ctx    硬件资源上下文 (分配给此节点的 InferContext)
     */
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};