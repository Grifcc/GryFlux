#pragma once

#include "framework/processing_task.h"
#include <vector>
#include <string>

namespace GryFlux
{
    class Classifier : public ProcessingTask
    {
    public:
        explicit Classifier(float threshold, const std::vector<std::string>& class_labels = {})
            : threshold_(threshold), class_labels_(class_labels) {}
        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;
    private:
        float threshold_;
        std::vector<std::string> class_labels_;
    };
}; 