#include "nodes/Input/InputNode.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& detect_packet = static_cast<DetectDataPacket&>(packet);
    (void)ctx;

    detect_packet.detections.clear();
    if (detect_packet.original_image.empty()) {
        LOG.error("[InputNode] Received empty frame for packet %d",
                  detect_packet.frame_id);
        detect_packet.markFailed();
    }
}

}  // namespace PipelineNodes
