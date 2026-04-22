#include "InferenceNode.h"

#include "context/infercontext.h"
#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<DeepLabPacket &>(packet);
    auto &npu = static_cast<InferContext &>(ctx);

    const size_t inputBytes = p.input_tensor.size() * sizeof(float);
    if (inputBytes != npu.getInputBufferSize())
    {
        throw std::runtime_error("InferenceNode: input_tensor size does not match model input buffer.");
    }

    if (npu.getNumOutputs() == 0)
    {
        throw std::runtime_error("InferenceNode: model has no outputs.");
    }

    npu.copyToDevice(p.input_tensor.data(), inputBytes);
    npu.executeModel();
    npu.copyToHost();

    const size_t outputBytes = npu.getOutputSize(0);
    if (outputBytes % sizeof(float) != 0)
    {
        throw std::runtime_error("InferenceNode: output buffer size is not float-aligned.");
    }

    const size_t outputFloatCount = outputBytes / sizeof(float);
    if (p.output_tensor.size() != outputFloatCount)
    {
        p.output_tensor.resize(outputFloatCount);
    }

    std::memcpy(
        p.output_tensor.data(),
        npu.getOutputHostBuffer(0),
        outputBytes);

    LOG.debug("Packet %d: inference done", p.frame_id);
}

} // namespace PipelineNodes
