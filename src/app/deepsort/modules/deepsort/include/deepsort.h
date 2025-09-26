#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "datatype.h"
#include "featuretensor.h"
#include "model.hpp"
#include "tracker.h"

class DeepSort
{
public:
    DeepSort(const std::string &modelPath, int batchSize, int featureDim, int cpu_id, rknn_core_mask npu_id);
    ~DeepSort();

    void sort(cv::Mat &frame, std::vector<DetectBox> &dets);

private:
    void sort(cv::Mat &frame, DETECTIONS &detections);
    void sort(cv::Mat &frame, DETECTIONSV2 &detectionsv2);
    void init();

    std::string enginePath;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    int maxBudget;
    float maxCosineDist;

    std::vector<RESULT_DATA> result;
    std::vector<std::pair<CLSCONF, DETECTBOX>> results;
    std::unique_ptr<tracker> objTracker;
    std::unique_ptr<FeatureTensor> featureExtractor1;
    std::unique_ptr<FeatureTensor> featureExtractor2;
    rknn_core_mask npu_id;
    int cpu_id;
};
