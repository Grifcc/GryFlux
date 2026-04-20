#include "nodes/Infer/InferNode.h"

#include "context/acl_infer_context.h"
#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <cstring>

namespace PipelineNodes {

void InferNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& resnet_packet = static_cast<ResNetPacket&>(packet);
    auto& infer_context = static_cast<resnet::AclInferContext&>(ctx);

    const size_t input_bytes =
        resnet_packet.preprocessed_data.size() * sizeof(float);
    if (input_bytes != infer_context.getInputBufferSize()) {
        LOG.error("[InferNode] Packet %llu input bytes=%zu, expected=%zu",
                  static_cast<unsigned long long>(resnet_packet.packet_id),
                  input_bytes,
                  infer_context.getInputBufferSize());
        resnet_packet.markFailed();
        return;
    }

    infer_context.copyToDevice(resnet_packet.preprocessed_data.data(), input_bytes);
    infer_context.executeModel();
    infer_context.copyToHost();

    const size_t output_float_count = infer_context.getOutputSize(0) / sizeof(float);
    if (resnet_packet.logits.size() != output_float_count) {
        resnet_packet.logits.resize(output_float_count);
    }

    std::memcpy(resnet_packet.logits.data(),
                infer_context.getOutputHostBuffer(0),
                output_float_count * sizeof(float));
}

}  // namespace PipelineNodes
