#include "PostprocessNode.h"
#include "../../packet/resnet_packet.h"
#include <vector>
#include <algorithm>

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) {
    auto &p = static_cast<ResNetPacket&>(packet);
    (void)ctx;

    std::vector<std::pair<float, int>> score_index_pairs(p.logits.size());
    for (size_t i = 0; i < p.logits.size(); ++i) {
        score_index_pairs[i] = {p.logits[i], (int)i};
    }
    
    std::partial_sort(
        score_index_pairs.begin(),
        score_index_pairs.begin() + 5,
        score_index_pairs.end(),
        std::greater<std::pair<float, int>>()
    );

    p.top1_class = score_index_pairs[0].second;
    p.top5_correct = false;
    for (int i = 0; i < 5; ++i) {
        if (score_index_pairs[i].second == p.ground_truth_label) {
            p.top5_correct = true;
            break;
        }
    }
}