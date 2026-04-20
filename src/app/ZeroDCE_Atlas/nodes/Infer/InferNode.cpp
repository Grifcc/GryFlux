#include "InferNode.h"

#include <cstring>
#include <stdexcept>

void InferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto& dce_packet = dynamic_cast<ZeroDcePacket&>(packet);
    auto& atlas_ctx = dynamic_cast<AtlasContext&>(ctx);

    const size_t input_size = dce_packet.input_tensor.size() * sizeof(float);
    if (input_size != atlas_ctx.getInputBufferSize()) {
        throw std::runtime_error("input tensor size mismatch");
    }

    atlas_ctx.copyToDevice(dce_packet.input_tensor.data(), input_size);
    atlas_ctx.executeModel();
    atlas_ctx.copyToHost();

    if (atlas_ctx.getNumOutputs() == 0) {
        throw std::runtime_error("model output count is zero");
    }

    const size_t output_size = atlas_ctx.getOutputSize(0);
    dce_packet.host_output_buffer.resize(output_size);
    std::memcpy(dce_packet.host_output_buffer.data(),
                atlas_ctx.getOutputHostBuffer(0),
                output_size);
}
