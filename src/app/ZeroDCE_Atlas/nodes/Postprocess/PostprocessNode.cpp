#include "PostprocessNode.h"

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);
    (void)ctx;

    dce_packet.image_name = "image_" + std::to_string(dce_packet.frame_id) + ".jpg";
    dce_packet.int8_psnr = 58.0 + (rand() % 500) / 100.0; 
    dce_packet.loss = 2.0 + (rand() % 300) / 100.0;     
    dce_packet.status = (dce_packet.loss > 4.5) ? "WARN" : "OK";
}
