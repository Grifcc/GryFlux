#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
#include <utility> // for std::pair
#include "datatype.h"

// 确保 common/package.h 已经定义了以下类型:
// using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
// using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
// using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
// using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
// using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
// using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;
// using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
// using DETECTBOXSS = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>;


/**
 * @brief 卡尔曼滤波器类，用于 DeepSORT 状态估计.
 * 状态空间: [cx, cy, a, h, vx, vy, va, vh]
 * (中心点x, y, 宽高比aspect ratio, 高度height, 以及对应的速度)
 * 测量空间: [cx, cy, a, h]
 */
class MyKalmanFilter {
public:
    static const double chi2inv95[10]; // 95% 置信度的卡方分布逆累积分布函数值

    MyKalmanFilter();

    /**
     * @brief 初始化卡尔曼滤波器状态.
     * @param measurement 初始检测框 [cx, cy, a, h].
     * @return 返回初始状态均值和协方差.
     */
    KAL_DATA initiate(const DETECTBOX& measurement);

    /**
     * @brief 执行预测步骤.
     * @param mean 当前状态均值 (输入/输出).
     * @param covariance 当前状态协方差 (输入/输出).
     */
    void predict(KAL_MEAN &mean, KAL_COVA &covariance);

    /**
     * @brief 将状态分布投影到测量空间.
     * @param mean 当前状态均值.
     * @param covariance 当前状态协方差.
     * @return 返回投影后的均值和协方差.
     */
    KAL_HDATA project(const KAL_MEAN &mean, const KAL_COVA &covariance);

    /**
     * @brief 执行更新步骤.
     * @param mean 当前状态均值 (预测后的).
     * @param covariance 当前状态协方差 (预测后的).
     * @param measurement 当前帧的测量值 [cx, cy, a, h].
     * @return 返回更新后的状态均值和协方差.
     */
    KAL_DATA update(const KAL_MEAN &mean, const KAL_COVA &covariance, const DETECTBOX &measurement);

    /**
     * @brief 计算状态和一组测量值之间的门控距离 (马氏距离平方).
     * @param mean 当前状态均值.
     * @param covariance 当前状态协方差.
     * @param measurements 一组测量值 [cx, cy, a, h].
     * @param only_position (当前未实现) 是否只考虑位置信息.
     * @return 返回一个行向量，每个元素是对应测量值的马氏距离平方.
     */
    Eigen::Matrix<float, 1, -1> gating_distance(
        const KAL_MEAN &mean,
        const KAL_COVA &covariance,
        const std::vector<DETECTBOX> &measurements,
        bool only_position = false);

private:
    Eigen::Matrix<float, 8, 8> _motion_mat;  // 状态转移矩阵 F
    Eigen::Matrix<float, 4, 8> _update_mat;  // 测量矩阵 H
    float _std_weight_position;          // 位置标准差权重
    float _std_weight_velocity;          // 速度标准差权重
};