#include "nodes/Output/OutputNode.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void OutputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    const auto& detect_packet = static_cast<const DetectDataPacket&>(packet);
    (void)ctx;

    LOG.info("[OutputNode] Packet %d completed with %zu detections",
             detect_packet.frame_id,
             detect_packet.detections.size());
}

}  // namespace PipelineNodes
