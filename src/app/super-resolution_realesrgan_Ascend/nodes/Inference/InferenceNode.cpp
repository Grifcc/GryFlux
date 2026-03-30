#include "InferenceNode.h"

#include "context/realesrgan_npu_context.h"
#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <cstring>
#include <stdexcept>

namespace PipelineNodes
{

void InferenceNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<RealEsrganPacket &>(packet);
    auto &npu = static_cast<RealEsrganNPUContext &>(ctx);

    if (p.lr_image.empty())
    {
        throw std::runtime_error("InferenceNode: lr_image is empty.");
    }

    npu.bindToCurrentThread();

    const size_t inputBytes = p.input_tensor.size() * sizeof(float);
    if (inputBytes > npu.getInputSize())
    {
        throw std::runtime_error("InferenceNode: input_tensor size exceeds model input buffer.");
    }

    p.output_buffer.resize(npu.getOutputSize());

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

    npu.refreshCurrentOutputDims();
    p.output_dims = npu.getCurrentOutputDims();
    p.output_format = npu.getOutputFormat();
    p.output_data_type = npu.getOutputDataType();

    CHECK_ACL(
        aclrtMemcpy(
            p.output_buffer.data(),
            p.output_buffer.size(),
            npu.getDevBufOut(),
            npu.getOutputSize(),
            ACL_MEMCPY_DEVICE_TO_HOST),
        "InferenceNode memcpy output");

    LOG.debug("Packet %d: inference done on device %d", p.frame_id, npu.getDeviceId());
}

} // namespace PipelineNodes
