#include "deepsort_tracker.h"

#include <vector>

#include "package.h"
#include "utils/logger.h"

namespace GryFlux
{
    DeepSortTracker::DeepSortTracker(const std::string &reid_model_path,
                                     int cpu_id,
                                     rknn_core_mask npu_id,
                                     int batch_size,
                                     int feature_dim)
    {
        tracker_ = std::make_unique<DeepSort>(reid_model_path, batch_size, feature_dim, cpu_id, npu_id);
    }

    std::shared_ptr<DataObject> DeepSortTracker::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.size() != 2)
        {
            LOG.error("DeepSortTracker expects 2 inputs but received %zu", inputs.size());
            return nullptr;
        }

        auto image_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        if (!image_data)
        {
            LOG.error("DeepSortTracker received invalid image input");
            return nullptr;
        }

        const int frame_id = image_data->get_id();
        auto detection_input = inputs[1];
        if (!detection_input)
        {
            return std::make_shared<TrackPackage>(frame_id);
        }

        auto object_data = std::dynamic_pointer_cast<ObjectPackage>(detection_input);
        if (!object_data)
        {
            LOG.error("DeepSortTracker received invalid detection input type");
            return std::make_shared<TrackPackage>(frame_id);
        }

        auto frame = image_data->get_data().clone();
        auto objects = object_data->get_data();

        std::vector<DetectBox> detections;
        detections.reserve(objects.size());
        for (const auto &obj : objects)
        {
            DetectBox box(static_cast<float>(obj.left),
                          static_cast<float>(obj.top),
                          static_cast<float>(obj.right),
                          static_cast<float>(obj.bottom),
                          obj.prob,
                          static_cast<float>(obj.class_id));
            detections.push_back(box);
        }

        tracker_->sort(frame, detections);

    const int track_frame_id = object_data->get_image_id();
    auto track_package = std::make_shared<TrackPackage>(track_frame_id >= 0 ? track_frame_id : frame_id);
        for (const auto &box : detections)
        {
            if (box.trackID < 0)
            {
                continue;
            }

            TrackedObject tracked{};
            tracked.left = static_cast<int>(box.x1);
            tracked.top = static_cast<int>(box.y1);
            tracked.right = static_cast<int>(box.x2);
            tracked.bottom = static_cast<int>(box.y2);
            tracked.class_id = static_cast<int>(box.classID);
            tracked.prob = box.confidence;
            tracked.track_id = static_cast<int>(box.trackID);
            track_package->push_data(tracked);
        }

        return track_package;
    }
}
