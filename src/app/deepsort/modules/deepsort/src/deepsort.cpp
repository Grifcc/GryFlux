#include "deepsort.h"

#include <algorithm>
#include <thread>

namespace
{
    constexpr int kNetInputChannel = 3;
}

DeepSort::DeepSort(const std::string &modelPath, int batchSizeIn, int featureDimIn, int cpu_id_in, rknn_core_mask npu_id_in)
    : enginePath(modelPath),
      batchSize(batchSizeIn),
      featureDim(featureDimIn),
      imgShape(128, 256),
      maxBudget(100),
      maxCosineDist(0.3f),
      objTracker(std::make_unique<tracker>(maxCosineDist, maxBudget)),
      featureExtractor1(std::make_unique<FeatureTensor>(enginePath.c_str(), cpu_id_in, npu_id_in, 1, 1)),
      featureExtractor2(std::make_unique<FeatureTensor>(enginePath.c_str(), cpu_id_in, npu_id_in, 1, 1)),
      npu_id(npu_id_in),
      cpu_id(cpu_id_in)
{
    init();
}

DeepSort::~DeepSort() = default;

void DeepSort::init()
{
    featureExtractor1->init(imgShape, featureDim, kNetInputChannel);
    featureExtractor2->init(imgShape, featureDim, kNetInputChannel);
}

void DeepSort::sort(cv::Mat &frame, std::vector<DetectBox> &dets)
{
    DETECTIONS detections;
    std::vector<CLSCONF> clsConf;
    detections.reserve(dets.size());
    clsConf.reserve(dets.size());

    for (const auto &det : dets)
    {
        DETECTBOX box(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        DETECTION_ROW row;
        row.tlwh = box;
        row.confidence = det.confidence;
        detections.push_back(row);
        clsConf.emplace_back(static_cast<int>(det.classID), det.confidence);
    }

    result.clear();
    results.clear();

    if (!detections.empty())
    {
        DETECTIONSV2 detectionsv2 = std::make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }

    std::vector<DetectBox> tracked;
    tracked.reserve(result.size());
    for (const auto &res : result)
    {
        const auto &b = res.second;
        DetectBox box(b(0), b(1), b(2) + b(0), b(3) + b(1), 1.0f);
        box.trackID = static_cast<float>(res.first);
        tracked.push_back(box);
    }

    for (std::size_t i = 0; i < tracked.size() && i < results.size(); ++i)
    {
        tracked[i].classID = static_cast<float>(results[i].first.cls);
        tracked[i].confidence = results[i].first.conf;
    }

    dets = std::move(tracked);
}

void DeepSort::sort(cv::Mat &frame, DETECTIONS &detections)
{
    if (!featureExtractor1->getRectsFeature(frame, detections))
    {
        return;
    }

    objTracker->predict();
    objTracker->update(detections);

    for (const auto &track : objTracker->tracks)
    {
        if (!track.is_confirmed() || track.time_since_update > 1)
        {
            continue;
        }
        result.emplace_back(track.track_id, track.to_tlwh());
    }
}

void DeepSort::sort(cv::Mat &frame, DETECTIONSV2 &detectionsv2)
{
    auto &detections = detectionsv2.second;

    const int numOfDetections = static_cast<int>(detections.size());
    bool flag1 = true;
    bool flag2 = true;

    if (numOfDetections < 2)
    {
        featureExtractor1->getRectsFeature(frame, detections);
    }
    else
    {
        DETECTIONS detectionsPart1;
        DETECTIONS detectionsPart2;
        int border = numOfDetections >> 1;
        detectionsPart1.assign(detections.begin(), detections.begin() + border);
        detectionsPart2.assign(detections.begin() + border, detections.end());

        std::thread reidThread1(&FeatureTensor::getRectsFeature, featureExtractor1.get(), std::ref(frame), std::ref(detectionsPart1));
        std::thread reidThread2(&FeatureTensor::getRectsFeature, featureExtractor2.get(), std::ref(frame), std::ref(detectionsPart2));
        reidThread1.join();
        reidThread2.join();

        for (int idx = 0; flag1 && flag2 && idx < numOfDetections; idx++)
        {
            if (idx < border)
            {
                detections[idx].updateFeature(detectionsPart1[idx].feature);
            }
            else
            {
                detections[idx].updateFeature(detectionsPart2[idx - border].feature);
            }
        }
    }

    if (flag1 && flag2)
    {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (const auto &track : objTracker->tracks)
        {
            if (!track.is_confirmed() || track.time_since_update > 1)
            {
                continue;
            }
            result.emplace_back(track.track_id, track.to_tlwh());
            results.emplace_back(CLSCONF(track.cls, track.conf), track.to_tlwh());
        }
    }
}