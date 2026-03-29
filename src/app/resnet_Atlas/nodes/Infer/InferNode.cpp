#include "InferNode.h"
#include "../../packet/resnet_packet.h"
#include "../../context/atlas_context.h"

void ResNetInferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    auto &atlasCtx = static_cast<AtlasContext&>(ctx);
    
    atlasCtx.executeInference(p.preprocessed_data, p.logits);
}