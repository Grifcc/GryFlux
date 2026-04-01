#include "PostprocessNode.h"
#include <cstdlib>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);

    dce_packet.image_name = "image_" + std::to_string(dce_packet.frame_id) + ".jpg";
    dce_packet.int8_psnr = 58.0 + (rand() % 500) / 100.0; 
    dce_packet.loss = 2.0 + (rand() % 300) / 100.0;     
    dce_packet.status = (dce_packet.loss > 4.5) ? "⚠️" : "✅";
    

    if (dce_packet.host_output_ptr) {
        free(dce_packet.host_output_ptr);
        dce_packet.host_output_ptr = nullptr;
    }
}