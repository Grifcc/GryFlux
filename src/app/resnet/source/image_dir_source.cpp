#include "source/image_dir_source.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>

ResnetImageDirSource::ResnetImageDirSource(const std::string &datasetDir)
    : datasetDir_(datasetDir)
{
    if (!fs::exists(datasetDir_) || !fs::is_directory(datasetDir_))
    {
        throw std::runtime_error("Invalid dataset directory: " + datasetDir_.string());
    }

    for (const auto &entry : fs::directory_iterator(datasetDir_))
    {
        if (!fs::is_regular_file(entry.status()))
        {
            continue;
        }

        if (isSupportedImage(entry.path()))
        {
            imageFiles_.push_back(entry.path());
        }
    }

    std::sort(imageFiles_.begin(), imageFiles_.end());
    setHasMore(!imageFiles_.empty());

    LOG.info("ResnetImageDirSource initialized: dir=%s files=%zu",
             datasetDir_.string().c_str(),
             imageFiles_.size());
}

std::unique_ptr<GryFlux::DataPacket> ResnetImageDirSource::produce()
{
    while (cursor_ < imageFiles_.size())
    {
        const auto imagePath = imageFiles_[cursor_++];
        cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (image.empty())
        {
            LOG.warning("Failed to read image, skip: %s", imagePath.string().c_str());
            continue;
        }

        auto packet = std::make_unique<ResnetPacket>();
        packet->idx = idx_++;
        packet->imagePath = imagePath.string();
        packet->originalImage = image;

        setHasMore(cursor_ < imageFiles_.size());
        return packet;
    }

    setHasMore(false);
    return nullptr;
}

bool ResnetImageDirSource::isSupportedImage(const fs::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}
