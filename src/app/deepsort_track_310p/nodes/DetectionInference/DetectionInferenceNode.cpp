#include "nodes/DetectionInference/DetectionInferenceNode.h"

#include "context/infercontext.h"
#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <cstring>

namespace PipelineNodes {

void DetectionInferenceNode::execute(
    GryFlux::DataPacket& packet,
    GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    infer_context.bindCurrentThread();

    const size_t input_bytes =
        track_packet.preproc_data.nchw_data.size() * sizeof(float);
    if (input_bytes != infer_context.getInputBufferSize()) {
        LOG.error("[DetectionInferenceNode] Packet %d input bytes=%zu, expected=%zu",
                  track_packet.frame_id,
                  input_bytes,
                  infer_context.getInputBufferSize());
        track_packet.markFailed();
        return;
    }

    infer_context.copyToDevice(track_packet.preproc_data.nchw_data.data(), input_bytes);
    infer_context.executeModel();
    infer_context.copyToHost();

    const size_t output_count = infer_context.getNumOutputs();
    if (track_packet.infer_outputs.size() < output_count) {
        track_packet.infer_outputs.resize(output_count);
    }

    for (size_t index = 0; index < output_count; ++index) {
        const size_t float_count = infer_context.getOutputSize(index) / sizeof(float);
        auto& output_buffer = track_packet.infer_outputs[index];
        if (output_buffer.size() != float_count) {
            output_buffer.resize(float_count);
        }

        std::memcpy(output_buffer.data(),
                    infer_context.getOutputHostBuffer(index),
                    float_count * sizeof(float));
    }
}

}  // namespace PipelineNodes
