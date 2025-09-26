#include "track_result_sender.h"

#include <array>
#include <iomanip>
#include <sstream>

#include "package.h"
#include "utils/logger.h"

namespace
{
    cv::Scalar toBgr(const cv::Vec3b &rgb)
    {
        return cv::Scalar(rgb[2], rgb[1], rgb[0]);
    }

    const std::array<cv::Vec3b, 12> kPalette = {
        cv::Vec3b(255, 99, 71),    // tomato
        cv::Vec3b(135, 206, 250),  // light sky blue
        cv::Vec3b(152, 251, 152),  // pale green
        cv::Vec3b(255, 215, 0),    // gold
        cv::Vec3b(218, 112, 214),  // orchid
        cv::Vec3b(100, 149, 237),  // cornflower blue
        cv::Vec3b(255, 182, 193),  // light pink
        cv::Vec3b(144, 238, 144),  // light green
        cv::Vec3b(255, 165, 0),    // orange
        cv::Vec3b(173, 216, 230),  // light blue
        cv::Vec3b(240, 230, 140),  // khaki
        cv::Vec3b(255, 228, 196)   // bisque
    };

    const std::vector<std::string> kClassLabels = {
        "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
        "potted plant", "bed", "dining table", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};

    cv::Scalar colorForId(int trackId)
    {
        if (trackId < 0)
        {
            return cv::Scalar(0, 255, 0);
        }
        std::size_t index = static_cast<std::size_t>(trackId) % kPalette.size();
        return toBgr(kPalette[index]);
    }

    std::string className(int classId)
    {
        if (classId >= 0 && classId < static_cast<int>(kClassLabels.size()))
        {
            return kClassLabels[classId];
        }
        return "cls" + std::to_string(classId);
    }
}

namespace GryFlux
{
    cv::Scalar TrackResultSender::pickColor(int trackId)
    {
        return colorForId(trackId);
    }

    std::string TrackResultSender::buildLabel(int classId, float confidence, int trackId)
    {
        std::ostringstream oss;
        if (trackId >= 0)
        {
            oss << "ID " << trackId << " | ";
        }
        oss << className(classId) << " " << std::fixed << std::setprecision(2) << confidence;
        return oss.str();
    }

    std::shared_ptr<DataObject> TrackResultSender::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.size() != 2)
        {
            LOG.error("[TrackResultSender] Expected 2 inputs, got %zu", inputs.size());
            return nullptr;
        }

        auto image_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
        if (!image_data)
        {
            LOG.error("[TrackResultSender] Invalid image input");
            return nullptr;
        }

        const int img_id = image_data->get_id();
        cv::Mat annotated = image_data->get_data().clone();

        std::vector<TrackedObject> tracked_objects;
        tracked_objects.reserve(32);

        if (auto track_pkg = std::dynamic_pointer_cast<TrackPackage>(inputs[1]))
        {
            tracked_objects = track_pkg->get_data();
        }
        else if (auto object_pkg = std::dynamic_pointer_cast<ObjectPackage>(inputs[1]))
        {
            auto objects = object_pkg->get_data();
            tracked_objects.reserve(objects.size());
            for (const auto &obj : objects)
            {
                TrackedObject converted{};
                converted.left = obj.left;
                converted.top = obj.top;
                converted.right = obj.right;
                converted.bottom = obj.bottom;
                converted.class_id = obj.class_id;
                converted.prob = obj.prob;
                converted.track_id = -1;
                tracked_objects.push_back(converted);
            }
        }
        else
        {
            LOG.error("[TrackResultSender] Unsupported payload type");
            return std::make_shared<ImagePackage>(annotated, img_id);
        }

        for (const auto &obj : tracked_objects)
        {
            cv::Rect bbox(cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom));
            cv::Scalar color = pickColor(obj.track_id);

            cv::rectangle(annotated, bbox, color, 2);

            std::string label = buildLabel(obj.class_id, obj.prob, obj.track_id);
            int baseLine = 0;
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::Rect textBg(bbox.x, std::max(bbox.y - textSize.height - 4, 0), textSize.width + 6, textSize.height + 4);
            cv::rectangle(annotated, textBg, color, cv::FILLED);
            cv::putText(annotated,
                        label,
                        cv::Point(textBg.x + 3, textBg.y + textSize.height + 1),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.5,
                        cv::Scalar(0, 0, 0),
                        1,
                        cv::LINE_AA);
        }

        return std::make_shared<ImagePackage>(annotated, img_id);
    }
}
