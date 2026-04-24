#include "InferNode.h"
#include "../../packet/resnet_packet.h"
#include "../../context/orin_context.h"

void ResNetInferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    if (!p.is_valid_image) {
        return;
    }

    auto &orinCtx = static_cast<OrinContext&>(ctx);
    orinCtx.executeInference(p.preprocessed_data, p.logits);
}
