#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <vector>

namespace PipelineNodes {

void PostprocessNode::execute(GryFlux::DataPacket& packet, GryFlux::Context& ctx) {
    auto& resnet_packet = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    if (resnet_packet.logits.empty()) {
        LOG.error("[PostprocessNode] Packet %llu has empty logits",
                  static_cast<unsigned long long>(resnet_packet.packet_id));
        resnet_packet.top1_class = -1;
        resnet_packet.top5_correct = false;
        resnet_packet.markFailed();
        return;
    }

    std::vector<std::pair<float, int>> score_index_pairs(resnet_packet.logits.size());
    for (size_t i = 0; i < resnet_packet.logits.size(); ++i) {
        score_index_pairs[i] = {resnet_packet.logits[i], static_cast<int>(i)};
    }

    const size_t top_k = std::min<size_t>(5, score_index_pairs.size());
    std::partial_sort(
        score_index_pairs.begin(),
        score_index_pairs.begin() + static_cast<ptrdiff_t>(top_k),
        score_index_pairs.end(),
        std::greater<std::pair<float, int>>());

    resnet_packet.top1_class = score_index_pairs[0].second;
    resnet_packet.top5_correct = false;
    for (size_t i = 0; i < top_k; ++i) {
        if (score_index_pairs[i].second == resnet_packet.ground_truth_label) {
            resnet_packet.top5_correct = true;
            break;
        }
    }
}

}  // namespace PipelineNodes
