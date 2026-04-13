#include "nodes/Preprocess/PreprocessNode.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

namespace PipelineNodes {

PreprocessNode::PreprocessNode(int model_width, int model_height)
    : model_width_(model_width),
      model_height_(model_height) {}

void PreprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    (void)ctx;

    if (fusion_packet.vis_raw_bgr.empty() || fusion_packet.ir_raw_gray.empty()) {
        LOG.error(
            "[PreprocessNode] Empty image pair for %s",
            fusion_packet.filename.c_str());
        fusion_packet.markFailed();
        return;
    }

    cv::resize(
        fusion_packet.vis_raw_bgr,
        fusion_packet.vis_resized_bgr,
        cv::Size(model_width_, model_height_));
    cv::resize(
        fusion_packet.ir_raw_gray,
        fusion_packet.ir_resized_gray,
        cv::Size(model_width_, model_height_));

    cv::cvtColor(
        fusion_packet.vis_resized_bgr,
        fusion_packet.vis_ycrcb,
        cv::COLOR_BGR2YCrCb);
    cv::extractChannel(fusion_packet.vis_ycrcb, fusion_packet.vis_y, 0);
    cv::extractChannel(fusion_packet.vis_ycrcb, fusion_packet.vis_cr, 1);
    cv::extractChannel(fusion_packet.vis_ycrcb, fusion_packet.vis_cb, 2);

    fusion_packet.vis_y.convertTo(
        fusion_packet.vis_y_float,
        CV_32FC1,
        1.0 / 255.0);
    fusion_packet.ir_resized_gray.convertTo(
        fusion_packet.ir_float,
        CV_32FC1,
        1.0 / 255.0);
}

}  // namespace PipelineNodes
