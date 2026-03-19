#include "reid_infer.h"
#include "../../packet/track_data_packet.h"
#include "../../context/reid_context.h"
#include <iostream>

ReidInferNode::ReidInferNode(int feat_dim) : feat_dim_(feat_dim) {}

void ReidInferNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    // 1. 强转数据包和硬件上下文
    auto &p = static_cast<TrackDataPacket&>(packet);
    auto &reid_ctx = static_cast<ReidContext&>(ctx);

    // 2. 硬件绑定
    // 确保当前线程对当前分配到的 NPU Device (0 或 1) 拥有控制权
    reid_ctx.bindCurrentThread();

    // 3. 清空并准备存放最终特征的车厢
    p.reid_features.clear();
    
    // 如果上一级预处理没切出图，说明这一帧没目标，直接下班
    if (p.reid_preproc_crops.empty()) {
        return;
    }

    // 4. 循环处理每一个目标（由于 ReID 模型 Batch 为 1，我们采用循环推理）
    for (const auto& crop_data : p.reid_preproc_crops) {
        
        // 如果是空数据（预处理失败的占位符），填入全 0 特征以保持索引一致
        if (crop_data.empty()) {
            p.reid_features.push_back(std::vector<float>(feat_dim_, 0.0f));
            continue;
        }

        // --- 核心 NPU 三部曲 ---
        
        // A. 拷贝：将 CPU 预处理好的 NCHW 数据搬运到 NPU 显存
        reid_ctx.copyToDevice(crop_data.data(), crop_data.size() * sizeof(float));

        // B. 推理：触发 NPU 硬件计算
        reid_ctx.execute();

        // C. 回传：将算好的 512 维向量拷回 CPU
        std::vector<float> feature = reid_ctx.copyToHost(feat_dim_);

        // 5. 将结果塞进 packet
        p.reid_features.push_back(std::move(feature));
    }
}