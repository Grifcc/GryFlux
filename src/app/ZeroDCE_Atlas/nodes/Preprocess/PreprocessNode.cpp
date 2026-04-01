#include <cstdlib>
#include "PreprocessNode.h"

void PreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);


    if (dce_packet.host_output_ptr) {
        free(dce_packet.host_output_ptr);
        dce_packet.host_output_ptr = nullptr;
    }
}