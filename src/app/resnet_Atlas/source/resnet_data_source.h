#pragma once

#include "framework/data_source.h"
#include "packet/resnet_packet.h"

#include <map>
#include <memory>
#include <string>

class ResNetDataSource : public GryFlux::DataSource {
public:
    ResNetDataSource(const std::string& dataset_dir,
                     const std::map<std::string, int>& gt_map);
    ~ResNetDataSource() override = default;

    std::unique_ptr<GryFlux::DataPacket> produce() override;

private:
    std::string dataset_dir_;
    std::map<std::string, int> gt_map_;
    std::map<std::string, int>::const_iterator it_;
    uint64_t current_id_ = 0;
};
