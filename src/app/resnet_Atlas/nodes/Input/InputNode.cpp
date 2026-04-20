#include "nodes/Input/InputNode.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InputNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& resnet_packet = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    if (resnet_packet.image_path.empty()) {
        LOG.error("[InputNode] Packet %llu has an empty image path",
                  static_cast<unsigned long long>(resnet_packet.packet_id));
        resnet_packet.markFailed();
    }
}

}  // namespace PipelineNodes
