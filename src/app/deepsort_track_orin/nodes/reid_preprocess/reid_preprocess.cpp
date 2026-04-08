#include "reid_preprocess.h"
#include "../../packet/track_data_packet.h"
#include <opencv2/opencv.hpp>

ReidPreprocessNode::ReidPreprocessNode(int target_w, int target_h)
    : target_w_(target_w), target_h_(target_h) {}

void ReidPreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    // 1. 类型转换与环境准备
    auto &p = static_cast<TrackDataPacket&>(packet);
    (void)ctx; // 纯 CPU 节点，无需 NPU Context

    // 清理并预分配中转车厢
    p.reid_preproc_crops.clear();
    if (p.detections.empty()) return;
    p.reid_preproc_crops.reserve(p.detections.size());

    // 2. 遍历所有检测到的目标进行“肢解”预处理
    for (const auto& det : p.detections) {
        
        // --- 步骤 A: 安全抠图 (Crop) ---
        // 必须进行边界检查，防止 YOLOX 预测的框超出原图边界导致 OpenCV 崩溃
        int x = std::max(0, static_cast<int>(det.x1));
        int y = std::max(0, static_cast<int>(det.y1));
        int w = std::min(p.original_image.cols - x, static_cast<int>(det.x2 - det.x1));
        int h = std::min(p.original_image.rows - y, static_cast<int>(det.y2 - det.y1));

        if (w <= 0 || h <= 0) {
            p.reid_preproc_crops.push_back({}); // 填入空数据占位，保持索引对应
            continue;
        }

        cv::Mat crop = p.original_image(cv::Rect(x, y, w, h));

        // --- 步骤 B: 尺寸调整 (Resize) ---
        cv::Mat resized;
        cv::resize(crop, resized, cv::Size(target_w_, target_h_));

        // --- 步骤 C: 标准化处理 (Normalize) ---
        // 转换 BGR 为 RGB，并归一化到 [0, 1]
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        // --- 步骤 D: HWC 转 NCHW ---
        // NPU 模型通常要求数据排列为：所有 R，接着所有 G，最后所有 B
        std::vector<float> nchw_data(3 * target_w_ * target_h_);
        std::vector<cv::Mat> channels(3);
        for (int i = 0; i < 3; ++i) {
            channels[i] = cv::Mat(target_h_, target_w_, CV_32FC1, nchw_data.data() + i * target_w_ * target_h_);
        }
        cv::split(resized, channels);

        // --- 步骤 E: 装车 ---
        p.reid_preproc_crops.push_back(std::move(nchw_data));
    }
}