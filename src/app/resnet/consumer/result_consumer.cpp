#include "consumer/result_consumer.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

ResnetResultConsumer::ResnetResultConsumer(const std::string &outputDir)
    : outputDir_(outputDir)
{
    if (outputDir_.empty())
    {
        throw std::runtime_error("Output directory is empty");
    }

    fs::create_directories(outputDir_);
    imagesDir_ = outputDir_ / "images";
    labelsDir_ = outputDir_ / "labels";
    fs::create_directories(imagesDir_);
    fs::create_directories(labelsDir_);

    LOG.info("ResnetResultConsumer image dir=%s", imagesDir_.string().c_str());
    LOG.info("ResnetResultConsumer label dir=%s", labelsDir_.string().c_str());
}

void ResnetResultConsumer::consume(std::unique_ptr<GryFlux::DataPacket> packet)
{
    if (!packet)
    {
        return;
    }

    consumedCount_.fetch_add(1, std::memory_order_relaxed);

    auto &p = static_cast<ResnetPacket &>(*packet);
    if (p.originalImage.empty())
    {
        LOG.warning("Frame idx=%d original image is empty", p.idx);
        return;
    }

    cv::Mat output = p.originalImage.clone();

    int y = 28;
    cv::putText(output, "ResNet Top-K", cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    y += 28;

    for (std::size_t i = 0; i < p.topK.size(); ++i)
    {
        const auto &result = p.topK[i];
        std::ostringstream line;
        line << "#" << (i + 1)
             << " " << result.label
             << " (" << result.classId << ") "
             << std::fixed << std::setprecision(2) << (result.probability * 100.0f) << "%";
        cv::putText(output, line.str(), cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2);
        y += 24;
    }

    std::ostringstream baseName;
    baseName << "resnet_" << std::setfill('0') << std::setw(6) << p.idx;

    const fs::path imagePath = imagesDir_ / (baseName.str() + ".jpg");
    const fs::path labelPath = labelsDir_ / (baseName.str() + ".txt");

    if (!cv::imwrite(imagePath.string(), output))
    {
        LOG.error("Failed to write result image: %s", imagePath.string().c_str());
        return;
    }

    std::ofstream labelFile(labelPath);
    if (!labelFile)
    {
        LOG.error("Failed to write result label: %s", labelPath.string().c_str());
        return;
    }
    for (std::size_t i = 0; i < p.topK.size(); ++i)
    {
        const auto &result = p.topK[i];
        labelFile << (i + 1) << ' '
                  << result.classId << ' '
                  << std::fixed << std::setprecision(6) << result.probability << ' '
                  << result.label << '\n';
    }

    writtenCount_.fetch_add(1, std::memory_order_relaxed);
}

