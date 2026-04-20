#include "InferNode.h"

#include <chrono>

void InferNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& dce_packet = static_cast<ZeroDcePacket&>(packet);
    if (!dce_packet.is_valid_image) {
        dce_packet.infer_ms = 0.0;
        return;
    }

    auto& orin_ctx = static_cast<OrinContext&>(ctx);
    const auto start_time = std::chrono::steady_clock::now();
    orin_ctx.executeInference(dce_packet.input_tensor, &dce_packet.output_tensor);
    const auto end_time = std::chrono::steady_clock::now();
    dce_packet.infer_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();
    dce_packet.status = "inferred";
}
