#include "nodes/Inference/InferenceNode.h"

#include "context/npu_context.h"
#include "packet/realesrgan_packet.h"

namespace RealesrganNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<RealesrganPacket &>(packet);
    auto &npu = static_cast<NpuContext &>(ctx);

    p.srTensorF32.release();
    npu.run(p.modelRgbU8, p.srTensorF32);
}

} // namespace RealesrganNodes
