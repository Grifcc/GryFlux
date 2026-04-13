#include "nodes/Inference/InferenceNode.h"

#include "context/infercontext.h"
#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InferenceNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    infer_context.bindCurrentThread();

    const size_t vis_element_count = fusion_packet.vis_y_float.total();
    const size_t ir_element_count = fusion_packet.ir_float.total();
    const size_t output_element_count = fusion_packet.fused_y_float.total();

    if (vis_element_count != infer_context.GetInputElementCount(0) ||
        ir_element_count != infer_context.GetInputElementCount(1) ||
        output_element_count != infer_context.GetOutputElementCount()) {
        LOG.error(
            "[InferenceNode] Tensor element count mismatch for %s",
            fusion_packet.filename.c_str());
        fusion_packet.markFailed();
        return;
    }

    infer_context.copyInputToDevice(
        0,
        reinterpret_cast<const float*>(fusion_packet.vis_y_float.data),
        vis_element_count);
    infer_context.copyInputToDevice(
        1,
        reinterpret_cast<const float*>(fusion_packet.ir_float.data),
        ir_element_count);
    infer_context.execute();
    infer_context.copyOutputToHost(
        reinterpret_cast<float*>(fusion_packet.fused_y_float.data),
        output_element_count);
}

}  // namespace PipelineNodes
