#include "PostprocessNode.h"
#include <cstdlib>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);

    dce_packet.image_name = "image_" + std::to_string(dce_packet.frame_id) + ".jpg";
    dce_packet.int8_psnr = 58.0 + (rand() % 500) / 100.0; // 模拟 58.00 ~ 63.00 的 PSNR
    dce_packet.loss = 2.0 + (rand() % 300) / 100.0;       // 模拟 2.00 ~ 5.00 的 Loss
    dce_packet.status = (dce_packet.loss > 4.5) ? "⚠️" : "✅";
    

    // 释放 CPU 内存，完全不需要理会 NPU
    if (dce_packet.host_output_ptr) {
        free(dce_packet.host_output_ptr);
        dce_packet.host_output_ptr = nullptr;
    }
}