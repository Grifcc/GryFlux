#pragma once

#include <vector>
#include <map>
#include <Eigen/Core>
#include "datatype.h" // 包含 FEATURESS, FEATURE, 等定义

// 确保 common/package.h 已经定义了以下类型:
// using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, 512, Eigen::RowMajor>;
// using FEATURE = Eigen::Matrix<float, 1, 512, Eigen::RowMajor>;
// 假设 TRACKER_DATA 定义为 std::pair<int, FEATURESS> (用于 partial_fit)
using TRACKER_DATA = std::pair<int, FEATURESS>;
// 假设 DYNAMICM 定义为 Eigen::MatrixXf
using DYNAMICM = Eigen::MatrixXf;

/**
 * @brief 计算最近邻距离度量，用于外观特征匹配.
 */
class NearNeighborDisMetric {
public:
    // 定义距离计算类型
    enum METRIC_TYPE {
        euclidean, // 欧氏距离 (示例代码中使用 _pdist 计算平方欧氏距离)
        cosine     // 余弦距离
    };

    /**
     * @brief 构造函数.
     * @param metric 使用的距离度量类型 (euclidean 或 cosine).
     * @param matching_threshold 匹配阈值. 对于余弦距离，通常小于 1 (例如 0.3); 对于欧氏距离，取决于特征范围.
     * @param budget 每个轨迹保留的最大特征样本数.
     */
    NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold, int budget);

    /**
     * @brief 计算当前帧检测到的特征与目标轨迹特征库之间的距离.
     * 对于每个目标轨迹，计算其特征库中所有样本与给定特征之间的最小距离 (或根据 metric 类型定义的其他聚合方式).
     * @param features 当前帧检测到的所有目标的特征 (MxN 矩阵, M 个目标, N 维特征).
     * @param targets 需要计算距离的目标轨迹 ID 列表.
     * @return 返回一个代价矩阵 (KxM), K 是 targets 的数量, M 是 features 的数量.
     * cost_matrix(i, j) 表示第 i 个 target 与第 j 个 feature 之间的最近邻距离.
     */
    DYNAMICM distance(const FEATURESS &features, const std::vector<int> &targets);

    /**
     * @brief 更新特征库 (samples).
     * @param tid_feats 一个包含 (track_id, features) 对的列表, features 是本帧匹配到该 track_id 的新特征.
     * @param active_targets 当前活跃 (未删除) 的轨迹 ID 列表.
     */
    void partial_fit(std::vector<TRACKER_DATA> &tid_feats, std::vector<int> &active_targets);

    // 公有成员变量
    float mating_threshold; // 匹配阈值

private:
    // 定义指向距离计算成员函数的指针类型
    using DistanceFunction = Eigen::VectorXf (NearNeighborDisMetric::*)(const FEATURESS&, const FEATURESS&);

    // 指向实际使用的距离计算函数 (_nncosine_distance 或 _nneuclidean_distance)
    DistanceFunction _metric;

    // 每个轨迹保留的最大特征样本数
    int budget;

    // 存储每个轨迹 ID 对应的特征样本库
    // key: track_id
    // value: FEATURESS (该轨迹的历史特征样本, 行数 <= budget)
    std::map<int, FEATURESS> samples;

    /**
     * @brief 计算 x 中每个样本与 y 中所有样本之间的余弦距离的最小值.
     * @param x 一个轨迹的历史特征样本 (KxN).
     * @param y 当前帧检测到的特征 (MxN).
     * @return 返回一个行向量 (1xM), 每个元素是 y 中对应特征与 x 中所有样本的最小余弦距离.
     */
    Eigen::VectorXf _nncosine_distance(const FEATURESS &x, const FEATURESS &y);

    /**
     * @brief 计算 x 中每个样本与 y 中所有样本之间的欧氏距离的最大值? (根据 cpp 实现似乎是最大值).
     * @param x 一个轨迹的历史特征样本 (KxN).
     * @param y 当前帧检测到的特征 (MxN).
     * @return 返回一个行向量 (1xM), 每个元素是 y 中对应特征与 x 中所有样本的最大欧氏距离(?).
     */
    Eigen::VectorXf _nneuclidean_distance(const FEATURESS &x, const FEATURESS &y);

    /**
     * @brief 计算两组特征向量之间的成对平方欧氏距离.
     * @param x 特征集 A (KxN).
     * @param y 特征集 B (MxN).
     * @return 返回一个距离矩阵 (KxM), distance(i, j) 是 x[i] 和 y[j] 之间的平方欧氏距离.
     */
    Eigen::MatrixXf _pdist(const FEATURESS &x, const FEATURESS &y);

    /**
     * @brief 计算两组特征向量之间的成对余弦距离 (1 - cosine_similarity).
     * @param a 特征集 A (KxN).
     * @param b 特征集 B (MxN).
     * @param data_is_normalized 指示输入特征向量是否已经 L2 归一化.
     * @return 返回一个距离矩阵 (KxM), distance(i, j) 是 a[i] 和 b[j] 之间的余弦距离.
     */
    Eigen::MatrixXf _cosine_distance(const FEATURESS &a, const FEATURESS &b, bool data_is_normalized = false);
};