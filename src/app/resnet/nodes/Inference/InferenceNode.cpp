#include "nodes/Inference/InferenceNode.h"

#include "context/resnet_npu_context.h"
#include "packet/resnet_packet.h"

namespace ResnetNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<ResnetPacket &>(packet);
    auto &npu = static_cast<ResnetNpuContext &>(ctx);

    npu.run(p.inputData, p.logits);
}

} // namespace ResnetNodes

