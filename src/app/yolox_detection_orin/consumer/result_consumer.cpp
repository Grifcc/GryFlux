#include "consumer/result_consumer.h"

#include "packet/detect_data_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

double NormalizeFps(double fps) {
    return fps > 0.0 ? fps : 25.0;
}

std::string BuildLabel(const Detection& detection) {
    std::ostringstream label;
    label << (detection.class_id == 0 ? "Person " : "Car ") << std::fixed
          << std::setprecision(2) << detection.score;
    return label.str();
}

bool IsVisualizedClass(int class_id) {
    return class_id == 0 || class_id == 2;
}

}  // namespace

ResultConsumer::ResultConsumer(
    const std::string& output_path,
    double fps,
    int width,
    int height)
    : writer_(output_path,
              cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
              NormalizeFps(fps),
              cv::Size(width, height)) {
    if (!writer_.isOpened()) {
        throw std::runtime_error("Failed to create output video: " + output_path);
    }

    LOG.info("[ResultConsumer] Writing annotated video to %s",
             output_path.c_str());
}

ResultConsumer::~ResultConsumer() {
    if (writer_.isOpened()) {
        writer_.release();
        LOG.info("[ResultConsumer] Output video finalized");
    }
}

void ResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet) {
    auto* detection_packet = static_cast<DetectDataPacket*>(packet.get());

    int rendered_count = 0;
    for (const auto& detection : detection_packet->detections) {
        if (!IsVisualizedClass(detection.class_id)) {
            continue;
        }

        cv::rectangle(
            detection_packet->original_image,
            cv::Point(static_cast<int>(detection.x1),
                      static_cast<int>(detection.y1)),
            cv::Point(static_cast<int>(detection.x2),
                      static_cast<int>(detection.y2)),
            cv::Scalar(0, 255, 0),
            2);
        cv::putText(
            detection_packet->original_image,
            BuildLabel(detection),
            cv::Point(static_cast<int>(detection.x1),
                      static_cast<int>(detection.y1) - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(0, 255, 0),
            1);
        ++rendered_count;
    }

    writer_.write(detection_packet->original_image);
    LOG.info("[ResultConsumer] Frame %d finished, rendered detections=%d",
             detection_packet->frame_id, rendered_count);
}
