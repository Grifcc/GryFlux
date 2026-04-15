#include "PostprocessNode.h"
#include "../../packet/resnet_packet.h"
#include <vector>
#include <algorithm>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    if (p.logits.empty()) {
        p.top1_class = -1;
        p.top5_correct = false;
        return;
    }

    std::vector<std::pair<float, int>> score_index_pairs(p.logits.size());
    for (size_t i = 0; i < p.logits.size(); ++i) {
        score_index_pairs[i] = {p.logits[i], (int)i};
    }

    size_t top_k = std::min<size_t>(5, score_index_pairs.size());
    
    std::partial_sort(
        score_index_pairs.begin(),
        score_index_pairs.begin() + top_k,
        score_index_pairs.end(),
        std::greater<std::pair<float, int>>()
    );

    p.top1_class = score_index_pairs[0].second;
    p.top5_correct = false;
    for (size_t i = 0; i < top_k; ++i) {
        if (score_index_pairs[i].second == p.ground_truth_label) {
            p.top5_correct = true;
            break;
        }
    }
}
