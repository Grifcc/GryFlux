#include "kalman_filter.h" // 修改包含路径
#include <Eigen/Cholesky> // Eigen 库内部通常会处理包含, 但显式包含更安全
#include <iostream>       // 用于打印错误信息
#include <vector>         // gating_distance 需要
#include "datatype.h"
// 95% 置信度的卡方分布逆累积分布函数值，索引对应自由度
const double MyKalmanFilter::chi2inv95[10] = {
    0,      // 0 degrees of freedom (unused)
    3.8415, // 1 degree of freedom
    5.9915, // 2 degrees of freedom
    7.8147, // 3 degrees of freedom
    9.4877, // 4 degrees of freedom
    11.070, // 5 degrees of freedom
    12.592, // 6 degrees of freedom
    14.067, // 7 degrees of freedom
    15.507, // 8 degrees of freedom
    16.919  // 9 degrees of freedom
};

MyKalmanFilter::MyKalmanFilter() {
    int ndim = 4; // 测量维度 (cx, cy, a, h)
    double dt = 1.0; // 时间步长

    // 初始化状态转移矩阵 F (8x8)
    _motion_mat = Eigen::Matrix<float, 8, 8>::Identity();
    for(int i = 0; i < ndim; i++) {
        _motion_mat(i, ndim + i) = static_cast<float>(dt);
    }

    // 初始化测量矩阵 H (4x8)
    _update_mat = Eigen::Matrix<float, 4, 8>::Identity();

    // 初始化噪声权重
    this->_std_weight_position = 1.0f / 20.0f;
    this->_std_weight_velocity = 1.0f / 160.0f;
}

KAL_DATA MyKalmanFilter::initiate(const DETECTBOX& measurement) {
    // measurement: [cx, cy, a, h]
    DETECTBOX mean_pos = measurement;
    DETECTBOX mean_vel = DETECTBOX::Zero(); // 初始速度设为 0

    // 初始化状态均值向量 mean (1x8)
    KAL_MEAN mean;
    mean.block<1, 4>(0, 0) = mean_pos;
    mean.block<1, 4>(0, 4) = mean_vel;

    // 初始化状态协方差矩阵 covariance (8x8)
    KAL_MEAN std_dev; // 用于存储标准差
    std_dev(0) = 2.0f * _std_weight_position * measurement(3); // cx
    std_dev(1) = 2.0f * _std_weight_position * measurement(3); // cy
    std_dev(2) = 1e-2f;                                        // a
    std_dev(3) = 2.0f * _std_weight_position * measurement(3); // h
    std_dev(4) = 10.0f * _std_weight_velocity * measurement(3); // vx
    std_dev(5) = 10.0f * _std_weight_velocity * measurement(3); // vy
    std_dev(6) = 1e-5f;                                        // va
    std_dev(7) = 10.0f * _std_weight_velocity * measurement(3); // vh

    // 方差 = 标准差的平方
    KAL_COVA covariance = std_dev.array().square().matrix().asDiagonal();
    return std::make_pair(mean, covariance);
}

void MyKalmanFilter::predict(KAL_MEAN &mean, KAL_COVA &covariance) {
    // 预测步骤:
    // x_pred = F * x
    // P_pred = F * P * F^T + Q

    // 1. 计算过程噪声协方差矩阵 Q
    DETECTBOX std_pos; // 位置标准差
    std_pos << _std_weight_position * mean(3), // cx
               _std_weight_position * mean(3), // cy
               1e-2f,                           // a
               _std_weight_position * mean(3);  // h
    DETECTBOX std_vel; // 速度标准差
    std_vel << _std_weight_velocity * mean(3), // vx
               _std_weight_velocity * mean(3), // vy
               1e-5f,                           // va
               _std_weight_velocity * mean(3);  // vh

    KAL_MEAN std_dev_combined; // 组合标准差向量
    std_dev_combined.block<1, 4>(0, 0) = std_pos;
    std_dev_combined.block<1, 4>(0, 4) = std_vel;

    // 过程噪声协方差 Q = 标准差平方构成的对角矩阵
    KAL_COVA motion_cov = std_dev_combined.array().square().matrix().asDiagonal();

    // 2. 预测状态均值: mean_pred = F * mean^T
    mean = (_motion_mat * mean.transpose()).transpose();

    // 3. 预测状态协方差: cov_pred = F * cov * F^T + Q
    covariance = _motion_mat * covariance * (_motion_mat.transpose());
    covariance += motion_cov;
}

KAL_HDATA MyKalmanFilter::project(const KAL_MEAN &mean, const KAL_COVA &covariance) {
    // 投影步骤: 将状态空间投影到测量空间
    // z_pred = H * x_pred
    // S = H * P_pred * H^T + R

    // 1. 计算测量噪声协方差矩阵 R
    DETECTBOX std_dev; // 测量标准差
    std_dev << _std_weight_position * mean(3), // cx
               _std_weight_position * mean(3), // cy
               1e-1f,                           // a
               _std_weight_position * mean(3);  // h

    // 测量噪声协方差 R = 标准差平方构成的对角矩阵
    KAL_HCOVA innovation_cov = std_dev.array().square().matrix().asDiagonal();

    // 2. 投影均值: mean_proj = H * mean^T
    KAL_HMEAN projected_mean = (_update_mat * mean.transpose()).transpose();

    // 3. 投影协方差 (Innovation Covariance S): S = H * cov * H^T + R
    // *** 修正点 1: 在使用 projected_cov 之前声明它 ***
    KAL_HCOVA projected_cov = _update_mat * covariance * (_update_mat.transpose());
    projected_cov += innovation_cov;

    return std::make_pair(projected_mean, projected_cov);
}

KAL_DATA
MyKalmanFilter::update(
        const KAL_MEAN &mean,         // 预测后的均值 x_pred
        const KAL_COVA &covariance,   // 预测后的协方差 P_pred
        const DETECTBOX &measurement) // 当前测量值 z
{
    // 更新步骤:
    // K = P_pred * H^T * S^{-1}
    // x_new = x_pred + K * (z - z_pred)
    // P_new = (I - K * H) * P_pred

    // 1. 投影到测量空间，得到 z_pred 和 S
    KAL_HDATA pa = project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;   // z_pred = H * x_pred (1x4)
    KAL_HCOVA projected_cov = pa.second;   // S = H * P_pred * H^T + R (4x4)

    // 2. 计算卡尔曼增益 K
    //   K = P_pred * H^T * S^{-1}
    //   解 S * K^T = (P_pred * H^T)^T = H * P_pred
    //   令 B = H * P_pred (维度 4x8)
    //   解 S * X = B (X 维度 4x8)
    //   则 K = X^T (维度 8x4)
    // *** 修正点 2: 重新计算 B ***
    Eigen::Matrix<float, 4, 8> B = _update_mat * covariance; // B = H * P_pred
    Eigen::Matrix<float, 8, 4> kalman_gain = (projected_cov.llt().solve(B)).transpose(); // K = (S^{-1} * B)^T = (S^{-1} * H * P)^T

    // 3. 计算测量残差 (Innovation): innovation = z - z_pred
    Eigen::Matrix<float, 1, 4> innovation = measurement - projected_mean; // 1x4

    // 4. 更新状态均值: new_mean = mean + innovation * K^T
    KAL_MEAN new_mean = mean + innovation * kalman_gain.transpose(); // (1x8) = (1x8) + (1x4) * (8x4)^T = (1x8) + (1x4)*(4x8)

    // 5. 更新状态协方差: new_cov = (I - K * H) * covariance
    KAL_COVA new_covariance = covariance - kalman_gain * _update_mat * covariance; // (8x8) = (8x8) - (8x4)*(4x8)*(8x8)

    return std::make_pair(new_mean, new_covariance);
}

Eigen::Matrix<float, 1, -1>
MyKalmanFilter::gating_distance(
        const KAL_MEAN &mean,              // 轨迹的预测状态均值
        const KAL_COVA &covariance,        // 轨迹的预测状态协方差
        const std::vector<DETECTBOX> &measurements, // 当前帧的所有检测框 [cx, cy, a, h]
        bool only_position)             // 是否只使用位置进行门控 (当前未实现)
{
    // 计算轨迹预测和一组测量值之间的马氏距离平方
    // d^2 = (z - z_pred)^T * S^{-1} * (z - z_pred)

    // 1. 投影到测量空间，得到 z_pred 和 S
    KAL_HDATA pa = this->project(mean, covariance);
    KAL_HMEAN projected_mean = pa.first;   // z_pred (1x4)
    KAL_HCOVA projected_cov = pa.second;   // S (4x4)

    if (only_position) {
        std::cerr << "gating_distance with only_position=true is not implemented!" << std::endl;
        return Eigen::Matrix<float, 1, -1>::Constant(1, measurements.size(), 1e+5f);
    }

    // 2. 计算每个测量值与预测值之间的差值 (z - z_pred)
    // 将 std::vector<DETECTBOX> 转换为 Eigen::Matrix (Nx4)
    DETECTBOXSS diff(measurements.size(), 4);
    for (size_t i = 0; i < measurements.size(); ++i) {
        diff.row(i) = measurements[i] - projected_mean; // measurements[i] (1x4), projected_mean (1x4)
    }

    // 3. 计算马氏距离平方 d^2 = diff * S^{-1} * diff^T
    // 使用 Cholesky 分解 S = L * L^T
    // 解 L * Y = diff^T (注意这里是 diff 的转置, 维度变为 4xN)
    // 则 Y = L^{-1} * diff^T
    // 马氏距离平方 d^2 的对角线元素 = (Y.array().square()).colwise().sum()
    Eigen::Matrix<float, 4, 4> L = projected_cov.llt().matrixL(); // Cholesky 分解 S = L * L^T (L 是 4x4 下三角矩阵)

    // 解 L * Y = diff^T。Eigen 的 solve 对于三角矩阵是高效的。
    // diff.transpose() 是 4xN 矩阵
    Eigen::Matrix<float, 4, -1> Y = L.triangularView<Eigen::Lower>().solve(diff.transpose());

    // 马氏距离平方 = Y 中每个列向量的平方和
    Eigen::Matrix<float, 1, -1> square_maha = Y.array().square().colwise().sum();

    return square_maha;
}