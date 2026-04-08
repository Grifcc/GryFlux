#include "deepsort_tracker.h"
#include "linear_assignment.h"
#include <algorithm>
#include <iostream>

// =========================================================================
// 1. 构造函数：初始化度量器和系统参数 (已修正初始化顺序警告)
// =========================================================================
DeepSortTracker::DeepSortTracker(float max_cosine_distance, int nn_budget, int max_age, int n_init)
    : metric_(NearNeighborDisMetric::cosine, max_cosine_distance, nn_budget),
      next_id_(1),
      max_cosine_distance_(max_cosine_distance),
      nn_budget_(nn_budget),
      max_age_(max_age),
      n_init_(n_init)
{
}

// =========================================================================
// 2. 核心 Update 接口
// =========================================================================
std::vector<Track> DeepSortTracker::update(const DETECTIONS& detections) {
    
    // --- 第一步：卡尔曼滤波预测下一帧位置 ---
    for (auto& track : tracks_) {
        track.predict(&kf_);
    }

    // --- 第二步：准备匹配数据 ---
    std::vector<int> confirmed_tracks;
    std::vector<int> unconfirmed_tracks;
    for (size_t i = 0; i < tracks_.size(); i++) {
        if (tracks_[i].is_confirmed()) {
            confirmed_tracks.push_back(i);
        } else {
            unconfirmed_tracks.push_back(i);
        }
    }

    std::vector<int> detection_indices;
    for (size_t i = 0; i < detections.size(); i++) {
        detection_indices.push_back(i);
    }

    // --- 第三步：级联匹配 (Cascade Matching) ---
    // 已经删除 la 实例化，直接使用单例 getInstance()
    TRACKER_MATCHD res_a = linear_assignment::getInstance()->matching_cascade(
        this, &DeepSortTracker::gated_metric,
        max_cosine_distance_, max_age_,
        tracks_, detections, confirmed_tracks, detection_indices);

    std::vector<std::pair<int, int>> matches_a = res_a.matches;
    std::vector<int> unmatched_tracks_a = res_a.unmatched_tracks;
    std::vector<int> unmatched_detections = res_a.unmatched_detections;

    // --- 第四步：IOU 匹配 (IOU Matching) ---
    std::vector<int> iou_track_candidates;
    for (int t : unmatched_tracks_a) {
        if (tracks_[t].time_since_update == 1) { 
            iou_track_candidates.push_back(t);
        }
    }
    for (int t : unconfirmed_tracks) {
        iou_track_candidates.push_back(t);
    }

    // 使用单例 getInstance()
    TRACKER_MATCHD res_b = linear_assignment::getInstance()->min_cost_matching(
        this, &DeepSortTracker::iou_cost,
        0.7f, 
        tracks_, detections, iou_track_candidates, unmatched_detections);

    std::vector<std::pair<int, int>> matches_b = res_b.matches;
    std::vector<int> unmatched_tracks_b = res_b.unmatched_tracks;
    std::vector<int> unmatched_detections_b = res_b.unmatched_detections;

    // --- 第五步：合并匹配结果 ---
    std::vector<std::pair<int, int>> matches = matches_a;
    matches.insert(matches.end(), matches_b.begin(), matches_b.end());

    // --- 第六步：轨迹状态更新 ---
    for (const auto& match : matches) {
        int track_idx = match.first;
        int det_idx = match.second;
        tracks_[track_idx].update(&kf_, detections[det_idx]);
    }

    std::vector<int> all_unmatched_tracks;
    for (int t : unmatched_tracks_a) {
        if (tracks_[t].time_since_update > 1) { 
            all_unmatched_tracks.push_back(t);
        }
    }
    all_unmatched_tracks.insert(all_unmatched_tracks.end(), unmatched_tracks_b.begin(), unmatched_tracks_b.end());

    for (int track_idx : all_unmatched_tracks) {
        if (tracks_[track_idx].state == TrackState::Tentative || tracks_[track_idx].time_since_update > max_age_) {
            tracks_[track_idx].state = TrackState::Deleted;
        }
    }

    for (int det_idx : unmatched_detections_b) {
        KAL_DATA data = kf_.initiate(detections[det_idx].to_xyah());
        tracks_.emplace_back(data.first, data.second, next_id_++, n_init_, max_age_, detections[det_idx].feature);
    }

    // --- 第七步：内存清理 ---
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
        [](const Track& t) { return t.is_deleted(); }), tracks_.end());

    // --- 第八步：更新特征度量器的特征库 ---
    std::vector<int> active_targets;
    std::vector<TRACKER_DATA> tid_features;
    for (const auto& track : tracks_) {
        if (!track.is_confirmed()) continue;
        active_targets.push_back(track.track_id);
        tid_features.push_back({track.track_id, track.features}); 
    }
    metric_.partial_fit(tid_features, active_targets);

    // --- 第九步：向流水线消费者返回活跃的、确定态的轨迹用于画图 ---
    std::vector<Track> active_tracks;
    for (const auto& track : tracks_) {
        if (track.is_confirmed() && track.time_since_update <= 1) {
            active_tracks.push_back(track);
        }
    }
    return active_tracks;
}

// =========================================================================
// 3. 距离度量回调函数
// =========================================================================

DYNAMICM DeepSortTracker::gated_metric(
    std::vector<Track>& tracks, 
    const DETECTIONS& detections, 
    const std::vector<int>& track_indices, 
    const std::vector<int>& detection_indices) 
{
    FEATURESS features(detection_indices.size(), 512);
    for (size_t i = 0; i < detection_indices.size(); i++) {
        features.row(i) = detections[detection_indices[i]].feature;
    }

    std::vector<int> targets;
    for (size_t i = 0; i < track_indices.size(); i++) {
        targets.push_back(tracks[track_indices[i]].track_id);
    }

    DYNAMICM cost_matrix = metric_.distance(features, targets);

    std::vector<DETECTBOX> measurements;
    for (int d_idx : detection_indices) {
        measurements.push_back(detections[d_idx].to_xyah());
    }
    
    for (size_t i = 0; i < track_indices.size(); i++) {
        auto& track = tracks[track_indices[i]];
        Eigen::Matrix<float, 1, -1> gate_dists = kf_.gating_distance(track.mean, track.covariance, measurements, false);
        
        for (size_t j = 0; j < detection_indices.size(); j++) {
            if (gate_dists(0, j) > 9.4877f) { 
                cost_matrix(i, j) = 100000.0f; 
            }
        }
    }

    return cost_matrix;
}

DYNAMICM DeepSortTracker::iou_cost(
    std::vector<Track>& tracks, 
    const DETECTIONS& detections, 
    const std::vector<int>& track_indices, 
    const std::vector<int>& detection_indices) 
{
    DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(track_indices.size(), detection_indices.size());
    
    for (size_t i = 0; i < track_indices.size(); i++) {
        auto track_box = tracks[track_indices[i]].to_tlwh();
        for (size_t j = 0; j < detection_indices.size(); j++) {
            auto det_box = detections[detection_indices[j]].tlwh;
            
            float ix = std::max(track_box(0), det_box(0));
            float iy = std::max(track_box(1), det_box(1));
            float iw = std::min(track_box(0) + track_box(2), det_box(0) + det_box(2)) - ix;
            float ih = std::min(track_box(1) + track_box(3), det_box(1) + det_box(3)) - iy;
            
            float iou = 0.0f;
            if (iw > 0 && ih > 0) {
                float intersection = iw * ih;
                float union_area = track_box(2) * track_box(3) + det_box(2) * det_box(3) - intersection;
                iou = intersection / union_area;
            }
            
            cost_matrix(i, j) = 1.0f - iou;
        }
    }
    return cost_matrix;
}