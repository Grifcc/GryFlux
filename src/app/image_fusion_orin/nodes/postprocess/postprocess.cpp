#include "postprocess.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <atomic>
#include <iostream>
#include <vector>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    // 1. 类型转换为业务数据包
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);

    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(fusion_packet.fused_y_float, &min_val, &max_val);

    enum class OutputRangeMode {
        kZeroToOne,
        kMinusOneToOne,
        kZeroTo255,
        kUnknown
    };

    OutputRangeMode range_mode = OutputRangeMode::kUnknown;
    if (min_val >= -1.1 && max_val <= 1.1) {
        range_mode = (min_val < -0.05) ? OutputRangeMode::kMinusOneToOne : OutputRangeMode::kZeroToOne;
    } else if (min_val >= -0.5 && max_val <= 255.5) {
        range_mode = OutputRangeMode::kZeroTo255;
    }

    switch (range_mode) {
        case OutputRangeMode::kZeroToOne:
            fusion_packet.fused_y_float.convertTo(fusion_packet.fused_y_uint8, CV_8UC1, 255.0);
            break;
        case OutputRangeMode::kMinusOneToOne:
            fusion_packet.fused_y_float.convertTo(fusion_packet.fused_y_uint8, CV_8UC1, 127.5, 127.5);
            break;
        case OutputRangeMode::kZeroTo255:
            fusion_packet.fused_y_float.convertTo(fusion_packet.fused_y_uint8, CV_8UC1);
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

    static std::atomic<int> log_count{0};
    const int current_log_index = log_count.fetch_add(1, std::memory_order_relaxed);
    if (current_log_index < 5) {
        const char* mode_name = "unknown";
        switch (range_mode) {
            case OutputRangeMode::kZeroToOne:
                mode_name = "[0,1]";
                break;
            case OutputRangeMode::kMinusOneToOne:
                mode_name = "[-1,1]";
                break;
            case OutputRangeMode::kZeroTo255:
                mode_name = "[0,255]";
                break;
            case OutputRangeMode::kUnknown:
                mode_name = "clipped";
                break;
        }
        std::cout << "[PostprocessNode] " << fusion_packet.filename
                  << " fused_y_float min=" << min_val
                  << " max=" << max_val
                  << " mode=" << mode_name << std::endl;
    }

    // 3. 通道合并 (组装 YCrCb)
    // 取出预处理阶段 (PreprocessNode) 保留在 Packet 中的原始颜色通道
    std::vector<cv::Mat> fused_channels = {
        fusion_packet.fused_y_uint8,  // Y 通道 (刚由 NPU 融合出来的)
        fusion_packet.vis_cr,         // Cr 通道 (原始可见光的)
        fusion_packet.vis_cb          // Cb 通道 (原始可见光的)
    };
    // 合并写入预分配的 fused_ycrcb 缓存中
    cv::merge(fused_channels, fusion_packet.fused_ycrcb);

    // 4. 颜色空间转换 (YCrCb -> BGR)
    // 转换结果写入预分配的 fused_result 中，这也是最终用于保存或显示的图像
    cv::cvtColor(fusion_packet.fused_ycrcb, fusion_packet.fused_result, cv::COLOR_YCrCb2BGR);

    // 此时融合算法全流程结束，该 Packet 可由框架继续推送给下游的 Consumer 节点进行落盘/推流
}
