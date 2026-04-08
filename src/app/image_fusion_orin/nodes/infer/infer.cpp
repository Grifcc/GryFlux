#include "infer.h"
#include "packet/fusion_data_packet.h"
#include "context/infercontext.h"
#include <iostream>

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);
    auto &infer_ctx = static_cast<InferContext&>(ctx);
    infer_ctx.bindCurrentThread();

    infer_ctx.copyInputToDevice(0, reinterpret_cast<const float*>(fusion_packet.vis_y_float.data), fusion_packet.vis_y_float.total());
    infer_ctx.copyInputToDevice(1, reinterpret_cast<const float*>(fusion_packet.ir_float.data), fusion_packet.ir_float.total());
    infer_ctx.execute();
    infer_ctx.copyOutputToHost(reinterpret_cast<float*>(fusion_packet.fused_y_float.data), fusion_packet.fused_y_float.total());
}
