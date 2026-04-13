#include "nodes/ReidInference/ReidInferenceNode.h"

#include "context/reid_context.h"
#include "packet/track_data_packet.h"
#include "utils/logger.h"

#include <algorithm>

namespace PipelineNodes {

ReidInferenceNode::ReidInferenceNode(int feature_dimension)
    : feature_dimension_(feature_dimension) {}

void ReidInferenceNode::execute(
    GryFlux::DataPacket& packet,
    GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    auto& reid_context = static_cast<ReidContext&>(ctx);

    reid_context.bindCurrentThread();
    track_packet.active_reid_feature_count = track_packet.active_reid_crop_count;

    if (track_packet.active_reid_crop_count == 0) {
        return;
    }

    const size_t expected_input_size = reid_context.getInputBufferSize();
    const size_t output_element_count = reid_context.getOutputElementCount();
    if (output_element_count < static_cast<size_t>(feature_dimension_)) {
        LOG.error(
            "[ReidInferenceNode] ReID context output elements=%zu, requested feature dimension=%d",
            output_element_count,
            feature_dimension_);
        track_packet.markFailed();
        return;
    }

    for (size_t index = 0; index < track_packet.active_reid_crop_count; ++index) {
        auto& feature = track_packet.reid_features[index];
        if (feature.size() != static_cast<size_t>(feature_dimension_)) {
            feature.resize(static_cast<size_t>(feature_dimension_), 0.0f);
        }

        if (!track_packet.reid_crop_valid_flags[index]) {
            std::fill(feature.begin(), feature.end(), 0.0f);
            continue;
        }

        const auto& crop_data = track_packet.reid_preproc_crops[index];
        const size_t crop_bytes = crop_data.size() * sizeof(float);
        if (crop_bytes != expected_input_size) {
            LOG.error("[ReidInferenceNode] Crop bytes=%zu, expected=%zu",
                      crop_bytes,
                      expected_input_size);
            track_packet.markFailed();
            return;
        }

        reid_context.copyToDevice(crop_data.data(), crop_bytes);
        reid_context.execute();
        reid_context.copyToHost(feature.data(), feature.size());
    }
}

}  // namespace PipelineNodes
