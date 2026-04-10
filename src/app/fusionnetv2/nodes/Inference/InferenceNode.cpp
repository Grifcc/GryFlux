#include "nodes/Inference/InferenceNode.h"

#include "context/fusion_npu_context.h"
#include "packet/fusionnetv2_packet.h"

namespace FusionNetV2Nodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<FusionNetV2Packet &>(packet);
    auto &npu = static_cast<FusionNpuContext &>(ctx);

    p.fusedYF32.release();
    npu.run(p.visYF32, p.infraredF32, p.fusedYF32);
}

} // namespace FusionNetV2Nodes
