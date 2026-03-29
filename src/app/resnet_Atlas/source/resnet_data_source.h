#pragma once
#include "framework/async_pipeline.h"
#include "packet/resnet_packet.h"
#include <map>
#include <string>
#include <memory>

class ResNetDataSource : public GryFlux::DataSource {
public:
    ResNetDataSource(const std::string& dataset_dir, const std::map<std::string, int>& gt_map)
        : dataset_dir_(dataset_dir), gt_map_(gt_map) {
        it_ = gt_map_.begin();
        setHasMore(it_ != gt_map_.end());
    }

    std::unique_ptr<GryFlux::DataPacket> produce() override {
        if (it_ == gt_map_.end()) {
            setHasMore(false);
            return nullptr;
        }

        auto packet = std::make_unique<ResNetPacket>();
        // 新增：每次产生数据包时，给它一个递增的唯一 ID
        packet->packet_id = current_id_++; 
        packet->image_path = dataset_dir_ + "/" + it_->first;
        packet->ground_truth_label = it_->second;

        ++it_;
        setHasMore(it_ != gt_map_.end());
        return packet;
    }

private:
    std::string dataset_dir_;
    std::map<std::string, int> gt_map_;
    std::map<std::string, int>::const_iterator it_;
    uint64_t current_id_ = 0; // 新增：用于生成递增的 ID
};