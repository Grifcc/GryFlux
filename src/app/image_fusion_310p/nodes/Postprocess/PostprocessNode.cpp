#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/fusion_data_packet.h"

#include <opencv2/opencv.hpp>

#include <vector>

namespace PipelineNodes {

void PostprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    (void)ctx;

    fusion_packet.fused_y_float.convertTo(fusion_packet.fused_y_uint8, CV_8UC1, 255.0);

    std::vector<cv::Mat> fused_channels = {
        fusion_packet.fused_y_uint8,
        fusion_packet.vis_cr,
        fusion_packet.vis_cb,
    };
    cv::merge(fused_channels, fusion_packet.fused_ycrcb);
    cv::cvtColor(fusion_packet.fused_ycrcb,
                 fusion_packet.fused_result,
                 cv::COLOR_YCrCb2BGR);
}

}  // namespace PipelineNodes
