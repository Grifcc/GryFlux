#include "nodes/Inference/InferenceNode.h"

#include "context/infercontext.h"
#include "packet/detect_data_packet.h"
#include "utils/logger.h"

#include <cstring>

namespace PipelineNodes {

void InferenceNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& detect_packet = static_cast<DetectDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    infer_context.bindCurrentThread();

    const size_t input_bytes =
        detect_packet.preproc_data.nchw_data.size() * sizeof(float);
    if (input_bytes != infer_context.getInputBufferSize()) {
        LOG.error(
            "[InferenceNode] Packet %d input bytes=%zu, expected=%zu",
            detect_packet.frame_id,
            input_bytes,
            infer_context.getInputBufferSize());
        detect_packet.markFailed();
        return;
    }

    infer_context.copyToDevice(
        detect_packet.preproc_data.nchw_data.data(),
        input_bytes);
    infer_context.executeModel();
    infer_context.copyToHost();

    const size_t output_count = infer_context.getNumOutputs();
    if (detect_packet.infer_outputs.size() < output_count) {
        detect_packet.infer_outputs.resize(output_count);
    }

    for (size_t index = 0; index < output_count; ++index) {
        const size_t float_count =
            infer_context.getOutputSize(index) / sizeof(float);
        auto& output_buffer = detect_packet.infer_outputs[index];
        if (output_buffer.size() != float_count) {
            output_buffer.resize(float_count);
        }

        std::memcpy(
            output_buffer.data(),
            infer_context.getOutputHostBuffer(index),
            float_count * sizeof(float));
    }
}

}  // namespace PipelineNodes
