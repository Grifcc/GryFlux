#include "source/image_dir_source.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <opencv2/opencv.hpp>

ImageDirSource::ImageDirSource(const std::string &datasetPath)
    : datasetPath_(datasetPath)
{
    if (!fs::exists(datasetPath_) || !fs::is_directory(datasetPath_))
    {
        throw std::runtime_error("Invalid dataset directory: " + datasetPath_.string());
    }

    for (const auto &entry : fs::directory_iterator(datasetPath_))
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

    LOG.info("ImageDirSource initialized: dir=%s files=%zu", datasetPath_.string().c_str(), imageFiles_.size());
    setHasMore(!imageFiles_.empty());
}

std::unique_ptr<GryFlux::DataPacket> ImageDirSource::produce()
{
    while (cursor_ < imageFiles_.size())
    {
        const auto filePath = imageFiles_[cursor_++];

        cv::Mat img = cv::imread(filePath.string(), cv::IMREAD_UNCHANGED);
        if (img.empty())
        {
            LOG.warning("Failed to read image, skip: %s", filePath.string().c_str());
            continue;
        }

        auto packet = std::make_unique<RealesrganPacket>();
        packet->idx = idx_++;
        packet->filename = filePath.filename().string();
        packet->inputBgrU8 = img;

        if (cursor_ >= imageFiles_.size())
        {
            setHasMore(false);
        }

        return packet;
    }

    setHasMore(false);
    return nullptr;
}

bool ImageDirSource::isSupportedImage(const fs::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c)
                   {
                       return static_cast<char>(std::tolower(c));
                   });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}
