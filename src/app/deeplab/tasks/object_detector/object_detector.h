#pragma once

#include "framework/processing_task.h"

namespace GryFlux
{
    class ObjectDetector : public ProcessingTask
    {
    public:
        explicit ObjectDetector(float threshold): threshold_(threshold){}
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;
    private:
        float threshold_;
    };
};

