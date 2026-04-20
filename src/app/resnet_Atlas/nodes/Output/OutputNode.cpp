#include "nodes/Output/OutputNode.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void OutputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    const auto& resnet_packet = static_cast<const ResNetPacket&>(packet);
    (void)ctx;

    LOG.info("[OutputNode] Packet %llu completed: top1=%d, gt=%d, top5_correct=%s",
             static_cast<unsigned long long>(resnet_packet.packet_id),
             resnet_packet.top1_class,
             resnet_packet.ground_truth_label,
             resnet_packet.top5_correct ? "true" : "false");
}

}  // namespace PipelineNodes
