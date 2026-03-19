#pragma once

#include <vector>
#include "datatype.h" // 引入我们刚写好的数学基石
/**
 * @brief 轨迹状态枚举
 */
enum class TrackState {
    Tentative = 1, // 暂定态（刚发现，需观察几帧）
    Confirmed = 2, // 确定态（正式追踪中）
    Deleted = 3    // 已删除（目标丢失过久）
};

class Track {
public:
    // --- 核心状态变量 ---
    KAL_MEAN mean;       // 卡尔曼滤波状态均值 (1x8)
    KAL_COVA covariance; // 卡尔曼滤波状态协方差 (8x8)
    
    int track_id;        // 唯一的轨迹 ID
    int hits;            // 累计匹配成功的次数
    int age;             // 轨迹自创建以来的总帧数
    int time_since_update; // 自上次匹配成功以来丢失的帧数
    TrackState state;    // 当前状态

    // 特征历史库：用于计算表观特征距离
    FEATURESS features; 

    // --- 构造与方法 ---
    Track(KAL_MEAN mean, KAL_COVA covariance, int track_id, 
          int n_init, int max_age, const FEATURE& feature);

    /**
     * @brief 预测：进入下一帧时，由卡尔曼滤波器预测新位置
     */
    void predict(class MyKalmanFilter* kf);

    /**
     * @brief 更新：匹配成功后，用新的检测结果修正卡尔曼状态
     */
    void update(class MyKalmanFilter* kf, const DETECTION_ROW& detection);
    /**
     * @brief 坐标转换：将 8 维状态向量转回检测框 [x, y, w, h]
     */
    DETECTBOX to_tlwh() const;

    bool is_confirmed() const { return state == TrackState::Confirmed; }
    bool is_deleted() const { return state == TrackState::Deleted; }

private:
    int n_init_;
    int max_age_;
};