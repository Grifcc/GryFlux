#include "nodes/Input/InputNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& track_packet = static_cast<TrackDataPacket&>(packet);
    (void)ctx;

    track_packet.ResetFrameState();

    if (track_packet.original_image.empty()) {
        LOG.error("[InputNode] Received empty frame for packet %d",
                  track_packet.frame_id);
        track_packet.markFailed();
    }
}

}  // namespace PipelineNodes
