#include "result_consumer.h"
#include <iostream>

ResultConsumer::ResultConsumer(const std::string& output_path, double fps, int width, int height) {
    // 1. 初始化 DeepSORT 追踪器
    tracker_ = std::make_unique<DeepSortTracker>(0.4f, 100);
    
    // 2. 初始化视频写入器
    writer_.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
}

ResultConsumer::~ResultConsumer() {
    if (writer_.isOpened()) {
        writer_.release();
    }
}

void ResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* p = static_cast<TrackDataPacket*>(packet.get());

    // --- 步骤 A: 放入缓冲区 ---
    // 无论谁先到，先按 frame_id 塞进 map 候车室
    reorder_buffer_[p->frame_id] = std::move(packet);

    // --- 步骤 B: 检查顺序，放行数据 ---
    // 只要候车室里有我们要的“下一帧”，就循环处理
    while (reorder_buffer_.count(expected_frame_id_) > 0) {
        
        // 1. 取出当前顺序正确的帧
        auto current_packet = std::move(reorder_buffer_[expected_frame_id_]);
        reorder_buffer_.erase(expected_frame_id_);
        
        // 2. 执行真正的业务逻辑
        processSequentialFrame(static_cast<TrackDataPacket*>(current_packet.get()));

        // 3. 期望值递增，寻找下一帧
        expected_frame_id_++;
    }
}

void ResultConsumer::processSequentialFrame(TrackDataPacket* p) {
    // 2. 将检测框和特征向量打包成算法需要的 DETECTIONS (std::vector<DETECTION_ROW>)
    DETECTIONS ds_input;

    // 【调试信息 1】：检查输入帧号和 NPU 送来的检测框数量
    // std::cout << "[DEBUG] Frame ID: " << p->frame_id 
    //           << " | NPU Detections: " << p->detections.size() << std::endl;
    
    for (size_t i = 0; i < p->detections.size(); ++i) {
        const auto& d = p->detections[i];
        // 假设 tlwh 格式为 [x1, y1, w, h]
        DETECTBOX box;
        box << d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1;
        
        // 获取对应的特征向量 (需要从 std::vector<float> 转为 Eigen::Matrix)
        FEATURE feat = Eigen::Map<FEATURE>(p->reid_features[i].data());
        
        ds_input.emplace_back(box, d.score, feat);
        // 【调试信息 2】打印前两个检测框的坐标和置信度，确认坐标反推是否正确
        // if (i < 2) {
        //     std::cout << "  -> Det " << i << ": [x:" << d.x1 << ", y:" << d.y1 
        //               << ", w:" << (d.x2 - d.x1) << ", h:" << (d.y2 - d.y1) 
        //               << "] score: " << d.score << std::endl;
        //             }
    }

    // 3. 调用单参数的 update
    p->active_tracks = tracker_->update(ds_input);
    // 【调试信息 3】：检查 Tracker 成功输出的确定态(Confirmed)轨迹数量
    // std::cout << "[DEBUG] Frame ID: " << p->frame_id 
    //           << " | Active Tracks Output: " << p->active_tracks.size() 
    //           << "\n----------------------------------------" << std::endl;

    // 4. 修正绘图逻辑 (使用 to_tlwh() 获取坐标)
    for (const auto& track : p->active_tracks) {
        auto tlwh = track.to_tlwh();
        cv::Rect rect(tlwh(0), tlwh(1), tlwh(2), tlwh(3));
        cv::rectangle(p->original_image, rect, cv::Scalar(0, 255, 0), 2);
        
        std::string label = "ID: " + std::to_string(track.track_id);
        cv::putText(p->original_image, label, cv::Point(rect.x, rect.y - 5), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }

    // 3. 写入文件
    if (writer_.isOpened()) {
        writer_.write(p->original_image);
    }

    if (p->frame_id % 30 == 0) {
        std::cout << "[Consumer] 已处理至第 " << p->frame_id << " 帧" << std::endl;
    }
}