#include "nodes/Input/InputNode.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    (void)ctx;

    if (fusion_packet.vis_raw_bgr.empty() || fusion_packet.ir_raw_gray.empty()) {
        LOG.error(
            "[InputNode] Empty image pair for %s",
            fusion_packet.filename.c_str());
        fusion_packet.markFailed();
    }
}

}  // namespace PipelineNodes
