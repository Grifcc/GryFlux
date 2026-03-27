#include "InferenceNode.h"

#include "context/deeplab_npu_Context.h"
#include "packet/deeplab_packet.h"
#include "utils/logger.h"

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<DeepLabPacket &>(packet);
    auto &npu = static_cast<DeepLabNPUContext &>(ctx);

    npu.bindToCurrentThread();

    const size_t inputBytes = p.input_tensor.size() * sizeof(float);
    const size_t outputBytes = p.output_tensor.size() * sizeof(float);
    if (inputBytes > npu.getInputSize())
    {
        throw std::runtime_error("InferenceNode: input_tensor size exceeds model input buffer.");
    }
    if (outputBytes > npu.getOutputSize())
    {
        throw std::runtime_error("InferenceNode: output_tensor size exceeds model output buffer.");
    }

    CHECK_ACL(
        aclrtMemcpy(
            npu.getDevBufIn(),
            npu.getInputSize(),
            p.input_tensor.data(),
            inputBytes,
            ACL_MEMCPY_HOST_TO_DEVICE),
        "InferenceNode memcpy input");

    CHECK_ACL(
        aclmdlExecute(
            npu.getModelId(),
            npu.getInputDataset(),
            npu.getOutputDataset()),
        "InferenceNode aclmdlExecute");

    CHECK_ACL(
        aclrtMemcpy(
            p.output_tensor.data(),
            outputBytes,
            npu.getDevBufOut(),
            outputBytes,
            ACL_MEMCPY_DEVICE_TO_HOST),
        "InferenceNode memcpy output");

    LOG.debug("Packet %d: inference done", p.frame_id);
}

} // namespace PipelineNodes
