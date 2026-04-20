#include "source/resnet_data_source.h"

#include "utils/logger.h"

#include <memory>

ResNetDataSource::ResNetDataSource(const std::string& dataset_dir,
                                   const std::map<std::string, int>& gt_map)
    : dataset_dir_(dataset_dir), gt_map_(gt_map) {
    it_ = gt_map_.begin();
    setHasMore(it_ != gt_map_.end());
    LOG.info("[ResNetDataSource] Loaded %zu images from %s",
             gt_map_.size(),
             dataset_dir_.c_str());
}

std::unique_ptr<GryFlux::DataPacket> ResNetDataSource::produce() {
    if (it_ == gt_map_.end()) {
        setHasMore(false);
        return nullptr;
    }

    auto packet = std::make_unique<ResNetPacket>();
    packet->packet_id = current_id_++;
    packet->image_path = dataset_dir_ + "/" + it_->first;
    packet->ground_truth_label = it_->second;

    ++it_;
    if (it_ == gt_map_.end()) {
        setHasMore(false);
        LOG.info("[ResNetDataSource] All %llu images dispatched",
                 static_cast<unsigned long long>(current_id_));
    }

    return packet;
}
