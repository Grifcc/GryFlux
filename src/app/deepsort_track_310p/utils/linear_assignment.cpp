#include "linear_assignment.h" // 包含修正后的头文件
#include "munkres.h"           // 包含 Munkres 类
#include "deepsort_tracker.h" // *** 修正点 1: 包含正确的头文件 ***
#include <map>
#include <vector>
#include <iostream>             // 用于调试输出 (如果需要)
#include <limits>               // 包含 std::numeric_limits
#include <cmath>                // 包含 std::isnan, std::isinf
// 定义 Matrix 类型 (如果 munkres.h 依赖的 matrix.h 没有全局定义)
// 你需要确保 Matrix 类可用
// #include "path/to/matrix.h" // 包含 matrix.h
// 假设 matrix.h 已经被 munkres.h 包含了

// --- 辅助函数：调用 Munkres 求解 ---
namespace { // 使用匿名命名空间限制作用域

// 将 Eigen 矩阵转换为 Munkres 使用的 Matrix<double>
// 并调用 Munkres 求解器
Eigen::Matrix<float, -1, 2, Eigen::RowMajor> solveHungarian(const DYNAMICM &cost_matrix) {
    const int rows = static_cast<int>(cost_matrix.rows());
    const int cols = static_cast<int>(cost_matrix.cols());

    if (rows == 0 || cols == 0) {
        return Eigen::Matrix<float, -1, 2, Eigen::RowMajor>(0, 2); // 返回空矩阵
    }

    // 确保 matrix.h 被包含且 Matrix 类可用
    Matrix<double> matrix(rows, cols);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
             // 检查 NaN 和无穷大
            if (std::isnan(cost_matrix(row, col)) || std::isinf(cost_matrix(row, col))) {
                // 用一个非常大的有限值替换，Munkres 类内部也会处理无穷大
                matrix(row, col) = std::numeric_limits<double>::max() / 2.0;
            } else {
                matrix(row, col) = static_cast<double>(cost_matrix(row, col));
            }
        }
    }

    // 调用 Munkres 求解器
    Munkres<double> solver;
    solver.solve(matrix); // matrix 会被修改

    // 提取匹配结果
    std::vector<std::pair<int, int>> pairs;
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            // Munkres 求解后, 0 表示匹配, -1 表示未匹配
            if (matrix(row, col) == 0) {
                pairs.emplace_back(row, col);
            }
        }
    }

    // 将结果转换为 Eigen 矩阵格式
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> result(static_cast<int>(pairs.size()), 2);
    for (int i = 0; i < result.rows(); ++i) {
        result(i, 0) = static_cast<float>(pairs[i].first);
        result(i, 1) = static_cast<float>(pairs[i].second);
    }
    return result;
}

} // 匿名命名空间结束

// --- linear_assignment 类的实现 ---

// 初始化静态实例指针
linear_assignment *linear_assignment::instance = nullptr;

// 私有构造函数
linear_assignment::linear_assignment() {}

// 获取单例实例
linear_assignment *linear_assignment::getInstance()
{
    if(instance == nullptr) { // 修正: 使用 nullptr 检查
         instance = new linear_assignment();
    }
    return instance;
}

TRACKER_MATCHD
linear_assignment::matching_cascade(
        DeepSortTracker *distance_metric, // *** 修正点 2: 参数类型 ***
        GATED_METRIC_FUNC distance_metric_func, // 函数指针类型已在头文件修正
        float max_distance,
        int cascade_depth,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int>& track_indices, // 需要匹配的轨迹索引 (通常是 confirmed)
        std::vector<int> detection_indices // 可选的输入检测索引 (通常为空, 在内部生成)
    )
{
    TRACKER_MATCHD res; // 最终结果

    // 如果未提供 detection_indices, 则使用所有 detections
    if (detection_indices.empty() && !detections.empty()) { // 检查 detections 是否为空
        detection_indices.resize(detections.size());
        for(size_t i = 0; i < detections.size(); i++) {
            detection_indices[i] = static_cast<int>(i);
        }
    }

    std::vector<int> unmatched_detections = detection_indices; // 初始时所有检测都未匹配
    res.matches.clear();
    std::map<int, int> matches_trackid; // 用于快速查找已匹配的轨迹 ID

    // 按 cascade_depth (轨迹丢失时间) 分层匹配
    for(int level = 0; level < cascade_depth; level++) {
        if(unmatched_detections.empty()) break; // 没有剩余的检测框了

        std::vector<int> track_indices_l; // 当前层级需要匹配的轨迹索引
        for(int k : track_indices) {
            // 检查索引是否有效
            if (k < 0 || static_cast<size_t>(k) >= tracks.size()) continue;
            // 匹配 time_since_update
            if(tracks[k].time_since_update == (level + 1)) {
                track_indices_l.push_back(k);
            }
        }

        if(track_indices_l.empty()) continue; // 当前层没有需要匹配的轨迹

        // 调用 min_cost_matching 进行当前层的匹配
        TRACKER_MATCHD tmp = min_cost_matching(
                    distance_metric, distance_metric_func, // 传递实例和函数指针
                    max_distance, tracks, detections, track_indices_l,
                    unmatched_detections); // 使用上一层剩下的未匹配检测

        // 更新未匹配的检测列表
        unmatched_detections = tmp.unmatched_detections;

        // 收集当前层匹配成功的对
        for(const auto& match : tmp.matches) {
            res.matches.push_back(match);
            matches_trackid[match.first] = match.second; // 记录轨迹已被匹配
        }
    }

    // 收集最终未匹配的检测
    res.unmatched_detections = unmatched_detections;

    // 收集最终未匹配的轨迹 (来自输入的 track_indices 中未出现在 matches_trackid 中的)
    for(int tidx : track_indices) {
         // 检查索引是否有效
        if (tidx < 0 || static_cast<size_t>(tidx) >= tracks.size()) continue;
        if(matches_trackid.find(tidx) == matches_trackid.end()) {
            res.unmatched_tracks.push_back(tidx);
        }
    }

    return res;
}

TRACKER_MATCHD
linear_assignment::min_cost_matching(
        DeepSortTracker *distance_metric, // *** 修正点 3: 参数类型 ***
        GATED_METRIC_FUNC distance_metric_func, // 函数指针类型已在头文件修正
        float max_distance,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices
    )
{
    TRACKER_MATCHD res;

    // 如果没有轨迹或没有检测，直接返回
    if(detection_indices.empty() || track_indices.empty()) {
        res.matches.clear();
        res.unmatched_tracks = track_indices;   // 所有轨迹都未匹配
        res.unmatched_detections = detection_indices; // 所有检测都未匹配
        return res;
    }

    // 1. 计算代价矩阵
    // 通过指向成员函数的指针调用 distance_metric_func (gated_metric 或 iou_cost)
    DYNAMICM cost_matrix = (distance_metric->*distance_metric_func)(
                tracks, detections, track_indices, detection_indices);

    // 2. 将超过最大距离的代价设为一个非常大的值 (INFTY_COST)
    for(int i = 0; i < cost_matrix.rows(); i++) {
        for(int j = 0; j < cost_matrix.cols(); j++) {
            float cost = cost_matrix(i,j);
            // 检查 NaN 或 Inf
            if (std::isnan(cost) || std::isinf(cost) || cost > max_distance) {
                 cost_matrix(i,j) = INFTY_COST;
            }
        }
    }

    // 3. 调用匈牙利算法求解
    Eigen::Matrix<float, -1, 2, Eigen::RowMajor> indices = solveHungarian(cost_matrix);

    // 4. 区分匹配和未匹配项
    res.matches.clear();
    res.unmatched_tracks.clear();
    res.unmatched_detections.clear();

    std::vector<bool> track_matched(track_indices.size(), false);
    std::vector<bool> detection_matched(detection_indices.size(), false);

    for(int i = 0; i < indices.rows(); i++) {
        int row = static_cast<int>(indices(i, 0));
        int col = static_cast<int>(indices(i, 1));

        // 边界检查
        if (row < 0 || static_cast<size_t>(row) >= track_indices.size() ||
            col < 0 || static_cast<size_t>(col) >= detection_indices.size()) {
            continue; // 无效索引
        }

        // 再次检查匹配的代价是否真的小于阈值
        if (row < cost_matrix.rows() && col < cost_matrix.cols() && cost_matrix(row, col) < max_distance) {
            int track_idx = track_indices[row];
            int detection_idx = detection_indices[col];
            res.matches.push_back(std::make_pair(track_idx, detection_idx));
            track_matched[row] = true;
            detection_matched[col] = true;
        }
    }

    // 收集未匹配的轨迹
    for(size_t i = 0; i < track_indices.size(); ++i) {
        if (!track_matched[i]) {
            res.unmatched_tracks.push_back(track_indices[i]);
        }
    }

    // 收集未匹配的检测
    for(size_t i = 0; i < detection_indices.size(); ++i) {
        if (!detection_matched[i]) {
            res.unmatched_detections.push_back(detection_indices[i]);
        }
    }

    return res;
}

DYNAMICM
linear_assignment::gate_cost_matrix(
        MyKalmanFilter *kf,           // 卡尔曼滤波器实例
        DYNAMICM &cost_matrix,        // 输入的代价矩阵 (会被修改)
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,    // cost_matrix 行对应的轨迹索引
        const std::vector<int> &detection_indices, // cost_matrix 列对应的检测索引
        float gated_cost,             // 用于替换超过门控阈值的代价
        bool only_position           // 是否只使用位置门控
    )
{
    // ... 函数体保持不变 ...
    int gating_dim = only_position ? 2 : 4;
    if (gating_dim <= 0 || gating_dim >= 10) {
         std::cerr << "Error: Invalid gating_dim in gate_cost_matrix: " << gating_dim << std::endl;
         return cost_matrix;
    }
    double gating_threshold = MyKalmanFilter::chi2inv95[gating_dim];

    std::vector<DETECTBOX> measurements;
    measurements.reserve(detection_indices.size());
    for(int idx : detection_indices) {
        if (idx < 0 || static_cast<size_t>(idx) >= detections.size()) continue;
        measurements.push_back(detections[idx].to_xyah());
    }
    if (measurements.empty() && !detection_indices.empty()) {
        cost_matrix.fill(gated_cost);
        return cost_matrix;
    }
     if (measurements.empty() && detection_indices.empty()) {
         return cost_matrix;
     }

    for(size_t i = 0; i < track_indices.size(); i++) {
        int track_idx = track_indices[i];
        if (track_idx < 0 || static_cast<size_t>(track_idx) >= tracks.size()) continue;
        Track& track = tracks[track_idx];

        Eigen::Matrix<float, 1, -1> gating_distance = kf->gating_distance(
                    track.mean, track.covariance, measurements, only_position);

        int valid_meas_idx = 0;
        for (size_t j = 0; j < detection_indices.size(); ++j) {
            int det_idx = detection_indices[j];
             if (det_idx < 0 || static_cast<size_t>(det_idx) >= detections.size()) continue; // 跳过无效检测索引

             // 简化假设：measurements 的顺序与有效的 detection_indices 顺序一致
             if (valid_meas_idx >= gating_distance.cols()) break;

            if (gating_distance(0, valid_meas_idx) > gating_threshold) {
                 cost_matrix(i, j) = gated_cost;
            }
            valid_meas_idx++;
        }
    }
    return cost_matrix; // 函数应该返回修改后的 cost_matrix
}