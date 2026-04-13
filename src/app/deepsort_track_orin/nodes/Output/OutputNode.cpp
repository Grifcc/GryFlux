#include "nodes/Output/OutputNode.h"

#include "packet/track_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void OutputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    const auto& track_packet = static_cast<const TrackDataPacket&>(packet);
    (void)ctx;

    LOG.info("[OutputNode] Packet %d completed with %zu detections and %zu features",
             track_packet.frame_id,
             track_packet.detections.size(),
             track_packet.active_reid_feature_count);
}

}  // namespace PipelineNodes
