#include <cstdlib>
#include "PreprocessNode.h"

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);

    // ... 你的后处理逻辑（比如保存图像等） ...

    // 释放 Preprocess 中分配的普通 CPU 内存
    if (dce_packet.host_output_ptr) {
        free(dce_packet.host_output_ptr);
        dce_packet.host_output_ptr = nullptr;
    }
}