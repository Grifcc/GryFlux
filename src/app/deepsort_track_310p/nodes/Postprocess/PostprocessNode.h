#pragma once

#include "framework/node_base.h"

#include <vector>

namespace PipelineNodes {

class PostprocessNode : public GryFlux::NodeBase {
public:
    PostprocessNode(
        int model_width,
        int model_height,
        float confidence_threshold = 0.3f,
        float nms_threshold = 0.45f);
    ~PostprocessNode() override = default;

    void execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) override;

private:
    int model_width_ = 0;
    int model_height_ = 0;
    float confidence_threshold_ = 0.3f;
    float nms_threshold_ = 0.45f;
    const int num_classes_ = 80;
    const std::vector<int> strides_ = {8, 16, 32};
};

}  // namespace PipelineNodes
