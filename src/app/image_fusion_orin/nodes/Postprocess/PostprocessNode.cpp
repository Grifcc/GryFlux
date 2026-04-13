#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/fusion_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <vector>

namespace {

enum class OutputRangeMode {
    kZeroToOne,
    kMinusOneToOne,
    kZeroTo255,
    kUnknown,
};

OutputRangeMode DetectOutputRangeMode(const cv::Mat& fused_y_float) {
    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(fused_y_float, &min_value, &max_value);

    if (min_value >= -1.1 && max_value <= 1.1) {
        return min_value < -0.05 ? OutputRangeMode::kMinusOneToOne
                                 : OutputRangeMode::kZeroToOne;
    }
    if (min_value >= -0.5 && max_value <= 255.5) {
        return OutputRangeMode::kZeroTo255;
    }
    return OutputRangeMode::kUnknown;
}

}  // namespace

namespace PipelineNodes {

void PostprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& fusion_packet = static_cast<FusionDataPacket&>(packet);
    (void)ctx;

    switch (DetectOutputRangeMode(fusion_packet.fused_y_float)) {
        case OutputRangeMode::kZeroToOne:
            fusion_packet.fused_y_float.convertTo(
                fusion_packet.fused_y_uint8,
                CV_8UC1,
                255.0);
            break;
        case OutputRangeMode::kMinusOneToOne:
            fusion_packet.fused_y_float.convertTo(
                fusion_packet.fused_y_uint8,
                CV_8UC1,
                127.5,
                127.5);
            break;
        case OutputRangeMode::kZeroTo255:
            fusion_packet.fused_y_float.convertTo(
                fusion_packet.fused_y_uint8,
                CV_8UC1);
            break;
        case OutputRangeMode::kUnknown:
        default: {
            cv::Mat clipped;
            cv::max(fusion_packet.fused_y_float, 0.0, clipped);
            cv::min(clipped, 255.0, clipped);
            clipped.convertTo(fusion_packet.fused_y_uint8, CV_8UC1);
            break;
        }
    }

    std::vector<cv::Mat> fused_channels = {
        fusion_packet.fused_y_uint8,
        fusion_packet.vis_cr,
        fusion_packet.vis_cb,
    };
    cv::merge(fused_channels, fusion_packet.fused_ycrcb);
    cv::cvtColor(
        fusion_packet.fused_ycrcb,
        fusion_packet.fused_result,
        cv::COLOR_YCrCb2BGR);
}

}  // namespace PipelineNodes
