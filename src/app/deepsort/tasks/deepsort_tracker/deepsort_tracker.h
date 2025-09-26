#pragma once

#include <memory>
#include <string>

#include "deepsort.h"
#include "runtime/rknn_api.h"
#include "framework/processing_task.h"

namespace GryFlux
{
    class DeepSortTracker : public ProcessingTask
    {
    public:
        DeepSortTracker(const std::string &reid_model_path,
                        int cpu_id,
                        rknn_core_mask npu_id,
                        int batch_size = 1,
                        int feature_dim = 512);

        std::shared_ptr<DataObject> process(const std::vector<std::shared_ptr<DataObject>> &inputs) override;

    private:
        std::unique_ptr<DeepSort> tracker_;
    };
}
