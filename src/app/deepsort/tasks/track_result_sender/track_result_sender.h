#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "framework/processing_task.h"

namespace GryFlux
{
    class TrackResultSender : public ProcessingTask
    {
    public:
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

    private:
        static cv::Scalar pickColor(int trackId);
        static std::string buildLabel(int classId, float confidence, int trackId);
    };
}
