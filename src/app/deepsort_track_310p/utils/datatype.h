#pragma once

#include <vector>
#include <utility>
#include <Eigen/Core>
#include <Eigen/Dense>

// --- Eigen 矩阵类型定义 ---
using FEATURE = Eigen::Matrix<float, 1, 512, Eigen::RowMajor>;           
using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, 512, Eigen::RowMajor>; 

using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;           
using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;           

using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;          
using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;  

using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;                         
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;          
using DETECTBOXSS = Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>; 

/**
 * @brief 算法内部使用的检测结果包装
 * 已包含：坐标 (tlwh)、置信度 (confidence) 和 特征向量 (feature)
 */
struct DETECTION_ROW {
    DETECTBOX tlwh;      // [top, left, width, height]
    float confidence;    
    FEATURE feature;     

    DETECTION_ROW(DETECTBOX t = DETECTBOX::Zero(), float c = 0.0f, FEATURE f = FEATURE::Zero()) 
        : tlwh(t), confidence(c), feature(f) {}

    // 格式转换：tlwh -> xyah (卡尔曼滤波标准输入)
    DETECTBOX to_xyah() const {
        DETECTBOX ret = tlwh;
        ret(0) += ret(2) / 2.0f; 
        ret(1) += ret(3) / 2.0f; 
        if (ret(3) != 0) ret(2) = ret(2) / ret(3); 
        else ret(2) = 0;
        return ret;
    }
};

using DETECTIONS = std::vector<DETECTION_ROW>;
