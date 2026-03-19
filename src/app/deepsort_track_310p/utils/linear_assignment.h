#pragma once

#include <vector>
#include <Eigen/Core>
#include "datatype.h"       // 包含 DETECTIONS, Track 等定义
#include "track.h"        // 包含 Track 类的定义
#include "kalman_filter.h"           // 包含 MyKalmanFilter 定义

// *** 修正点 1: 确保包含 Eigen Matrix 定义 ***
#include <Eigen/Dense> // 包含 MatrixXf

// *** 修正点 2: 定义 DYNAMICM (放在使用之前) ***
using DYNAMICM = Eigen::MatrixXf; // 或者 Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>

// 假设 common/package.h 定义了:
// using DETECTIONS = std::vector<DETECTION_ROW>;
using MATCH_DATA = std::pair<int, int>; // 定义匹配对类型

struct TRACKER_MATCHD {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

// 前向声明 DeepSortTracker 类
class DeepSortTracker;

/**
 * @brief 使用匈牙利算法进行线性分配 (匹配).
 * 这是一个单例类.
 */
class linear_assignment {
public:
    // 定义代价计算函数的指针类型 (现在 DYNAMICM 已定义)
    using GATED_METRIC_FUNC = DYNAMICM (DeepSortTracker::*)(std::vector<Track>&, const DETECTIONS&, const std::vector<int>&, const std::vector<int>&);

    // 获取单例实例
    static linear_assignment* getInstance();

    /**
     * @brief 级联匹配.
     */
    TRACKER_MATCHD matching_cascade(
        DeepSortTracker *distance_metric,
        GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        int cascade_depth,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int>& track_indices,
        std::vector<int> detection_indices = {}
    );


    /**
     * @brief 最小代价匹配 (匈牙利算法).
     */
    TRACKER_MATCHD min_cost_matching(
        DeepSortTracker *distance_metric,
        GATED_METRIC_FUNC distance_metric_func,
        float max_distance,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        std::vector<int> &track_indices,
        std::vector<int> &detection_indices
    );

    /**
     * @brief 对代价矩阵应用门控. (*** 修正点 3: 返回类型 DYNAMICM ***)
     */
    DYNAMICM gate_cost_matrix(
        MyKalmanFilter *kf,
        DYNAMICM &cost_matrix,
        std::vector<Track> &tracks,
        const DETECTIONS &detections,
        const std::vector<int> &track_indices,
        const std::vector<int> &detection_indices,
        float gated_cost = 10000.0f,
        bool only_position = false
    );


private:
    linear_assignment(); // 私有构造函数 (单例)
    static linear_assignment *instance; // 静态实例指针
};

// 定义一个表示无穷大成本的值
const float INFTY_COST = 10000.0f;