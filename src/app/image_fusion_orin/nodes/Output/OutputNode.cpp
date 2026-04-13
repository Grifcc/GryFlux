#include "nodes/Output/OutputNode.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void OutputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    const auto& fusion_packet = static_cast<const FusionDataPacket&>(packet);
    (void)ctx;

    if (fusion_packet.packet_idx % 50 == 0) {
        LOG.info(
            "[OutputNode] Packet %llu completed for %s",
            static_cast<unsigned long long>(fusion_packet.packet_idx),
            fusion_packet.filename.c_str());
    }
}

}  // namespace PipelineNodes
