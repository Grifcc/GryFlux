#include "nodes/Inference/InferenceNode.h"

#include "context/infercontext.h"
#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes {

void InferenceNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    auto& infer_context = static_cast<InferContext&>(ctx);

    infer_context.bindCurrentThread();

    const size_t vis_elements =
        static_cast<size_t>(fusion_packet.vis_y_float.rows) *
        static_cast<size_t>(fusion_packet.vis_y_float.cols);
    const size_t ir_elements =
        static_cast<size_t>(fusion_packet.ir_float.rows) *
        static_cast<size_t>(fusion_packet.ir_float.cols);
    const size_t output_elements =
        static_cast<size_t>(fusion_packet.fused_y_float.rows) *
        static_cast<size_t>(fusion_packet.fused_y_float.cols);

    if (vis_elements != infer_context.GetInputElementCount(0) ||
        ir_elements != infer_context.GetInputElementCount(1) ||
        output_elements != infer_context.GetOutputElementCount()) {
        LOG.error("[InferenceNode] Model IO size mismatch for packet %llu",
                  static_cast<unsigned long long>(fusion_packet.packet_idx));
        fusion_packet.markFailed();
        return;
    }

    infer_context.copyInputToDevice(0,
                                    fusion_packet.vis_y_float.ptr<float>(),
                                    vis_elements);
    infer_context.copyInputToDevice(1,
                                    fusion_packet.ir_float.ptr<float>(),
                                    ir_elements);
    infer_context.execute();
    infer_context.copyOutputToHost(fusion_packet.fused_y_float.ptr<float>(),
                                   output_elements);
}

}  // namespace PipelineNodes
