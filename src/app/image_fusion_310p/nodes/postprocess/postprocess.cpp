#include "postprocess.h"
#include "packet/fusion_data_packet.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    // 1. 类型转换为业务数据包
    auto &fusion_packet = static_cast<FusionDataPacket&>(packet);

    // ==========================================================
    // 开始后处理：全程复用 Packet 中预先分配的 cv::Mat 缓存
    // ==========================================================

    // 2. 反归一化 (Float32 -> Uint8，数值乘以 255.0)
    // fused_y_uint8 在 Packet 构造时已分配好 CV_8UC1 的内存，此处直接覆写
    fusion_packet.fused_y_float.convertTo(fusion_packet.fused_y_uint8, CV_8UC1, 255.0);

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