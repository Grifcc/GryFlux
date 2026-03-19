#include "track.h"
#include "kalman_filter.h"
#include <iostream> // 引入调试打印

Track::Track(KAL_MEAN mean, KAL_COVA covariance, int track_id, 
             int n_init, int max_age, const FEATURE& feature)
    : mean(mean), covariance(covariance), track_id(track_id),
      hits(1), age(1), time_since_update(0), state(TrackState::Tentative),
      n_init_(n_init), max_age_(max_age) {
    
    // 初始化特征库
    features = feature; 
}

void Track::predict(MyKalmanFilter* kf) {
    // 【调试】打印预测前的中心点 X 坐标
    // std::cout << "[DEBUG] Track " << track_id << " 准备 Predict. 当前 cx: " << mean(0) << std::endl;
    
    // predict 是 void，在原地修改状态
    kf->predict(this->mean, this->covariance);
    
    this->age++;
    this->time_since_update++;
}

void Track::update(MyKalmanFilter* kf, const DETECTION_ROW& detection) {
    // std::cout << "[DEBUG] Track " << track_id << " 准备 Update. 匹配前 cx: " << mean(0) << std::endl;

    // 【致命修复】：update 有返回值 KAL_DATA，必须接收并覆盖当前的 mean 和 covariance！
    KAL_DATA res = kf->update(this->mean, this->covariance, detection.to_xyah());
    this->mean = res.first;
    this->covariance = res.second;

    // std::cout << "[DEBUG] Track " << track_id << " 完成 Update. 匹配后新 cx: " << mean(0) << "\n" << std::endl;

    this->hits++;
    this->time_since_update = 0;
    if (this->state == TrackState::Tentative && this->hits >= n_init_) {
        this->state = TrackState::Confirmed;
    }
}

DETECTBOX Track::to_tlwh() const {
    // 状态向量格式为 [cx, cy, a, h]，转回 [left, top, w, h]
    DETECTBOX ret = mean.leftCols(4);
    ret(2) *= ret(3); // width = aspect_ratio * height
    ret(0) -= ret(2) / 2.0f; // left = cx - w/2
    ret(1) -= ret(3) / 2.0f; // top = cy - h/2
    return ret;
}