#include "source/fusion_image_pair_source.h"

#include "packet/fusionnetv2_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace
{
constexpr const char *kVisibleDirName = "visible";
constexpr const char *kInfraredDirName = "infrared";
}

FusionImagePairSource::FusionImagePairSource(const std::string &datasetRoot)
    : datasetRoot_(datasetRoot),
      visibleDir_(datasetRoot_ / kVisibleDirName),
      infraredDir_(datasetRoot_ / kInfraredDirName)
{
    if (!fs::exists(datasetRoot_) || !fs::is_directory(datasetRoot_))
    {
        throw std::runtime_error("Invalid dataset root: " + datasetRoot_.string());
    }
    if (!fs::exists(visibleDir_) || !fs::is_directory(visibleDir_))
    {
        throw std::runtime_error("Visible directory not found: " + visibleDir_.string());
    }
    if (!fs::exists(infraredDir_) || !fs::is_directory(infraredDir_))
    {
        throw std::runtime_error("Infrared directory not found: " + infraredDir_.string());
    }

    for (const auto &entry : fs::directory_iterator(visibleDir_))
    {
        if (!fs::is_regular_file(entry.status()) || !isSupportedImage(entry.path()))
        {
            continue;
        }

        const std::string filename = entry.path().filename().string();
        const fs::path infraredPath = infraredDir_ / filename;
        if (!fs::exists(infraredPath) || !fs::is_regular_file(infraredPath))
        {
            LOG.warning("FusionImagePairSource skip unmatched visible image: %s", entry.path().string().c_str());
            continue;
        }

        filenames_.push_back(filename);
    }

    std::sort(filenames_.begin(), filenames_.end());
    setHasMore(!filenames_.empty());

    LOG.info("FusionImagePairSource initialized: root=%s pairs=%zu",
             datasetRoot_.string().c_str(),
             filenames_.size());
}

std::unique_ptr<GryFlux::DataPacket> FusionImagePairSource::produce()
{
    while (cursor_ < filenames_.size())
    {
        const std::string &filename = filenames_[cursor_++];
        const fs::path visiblePath = visibleDir_ / filename;
        const fs::path infraredPath = infraredDir_ / filename;

        cv::Mat visible = cv::imread(visiblePath.string(), cv::IMREAD_COLOR);
        cv::Mat infrared = cv::imread(infraredPath.string(), cv::IMREAD_GRAYSCALE);

        if (visible.empty() || infrared.empty())
        {
            LOG.warning("FusionImagePairSource failed to read pair: %s / %s",
                        visiblePath.string().c_str(),
                        infraredPath.string().c_str());
            continue;
        }

        auto packet = std::make_unique<FusionNetV2Packet>();
        packet->idx = idx_++;
        packet->filename = filename;
        packet->visibleBgrU8 = visible;
        packet->infraredGrayU8 = infrared;

        if (cursor_ >= filenames_.size())
        {
            setHasMore(false);
        }

        return packet;
    }

    setHasMore(false);
    return nullptr;
}

bool FusionImagePairSource::isSupportedImage(const fs::path &path)
{
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char ch)
                   {
                       return static_cast<char>(std::tolower(ch));
                   });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}
