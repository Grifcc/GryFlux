#include "nodes/Inference/InferenceNode.h"

#include "context/deeplab_npu_context.h"
#include "packet/deeplab_packet.h"

namespace DeeplabNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<DeeplabPacket &>(packet);
    auto &npu = static_cast<DeeplabNpuContext &>(ctx);

    npu.run(p.inputData, p.inferenceOutputs);
}

} // namespace DeeplabNodes
