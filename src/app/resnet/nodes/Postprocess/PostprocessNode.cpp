#include "nodes/Postprocess/PostprocessNode.h"

#include "packet/resnet_packet.h"
#include "utils/logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ResnetNodes
{

void PostprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<ResnetPacket &>(packet);

    if (p.logits.empty())
    {
        throw std::runtime_error("Postprocess: logits are empty");
    }

    const float maxLogit = *std::max_element(p.logits.begin(), p.logits.end());
    float sum = 0.0f;
    for (auto &value : p.logits)
    {
        value = std::exp(value - maxLogit);
        sum += value;
    }
    if (sum <= 0.0f)
    {
        throw std::runtime_error("Postprocess: invalid softmax sum");
    }
    for (auto &value : p.logits)
    {
        value /= sum;
    }

    if (p.sortedIndices.size() != p.logits.size())
    {
        p.sortedIndices.resize(p.logits.size());
    }
    std::iota(p.sortedIndices.begin(), p.sortedIndices.end(), 0);

    const std::size_t topKLimit = std::min(topKCount_, p.sortedIndices.size());
    std::partial_sort(p.sortedIndices.begin(),
                      p.sortedIndices.begin() + static_cast<std::ptrdiff_t>(topKLimit),
                      p.sortedIndices.end(),
                      [&p](std::size_t lhs, std::size_t rhs)
                      {
                          return p.logits[lhs] > p.logits[rhs];
                      });

    p.topK.clear();
    if (p.topK.capacity() < topKLimit)
    {
        p.topK.reserve(topKLimit);
    }

    for (std::size_t i = 0; i < topKLimit; ++i)
    {
        const std::size_t classId = p.sortedIndices[i];
        std::string label = "class_" + std::to_string(classId);
        if (classId < classLabels_.size() && !classLabels_[classId].empty())
        {
            label = classLabels_[classId];
        }

        p.topK.push_back({
            static_cast<int>(classId),
            p.logits[classId],
            std::move(label),
        });
    }

    if (!p.topK.empty())
    {
        const auto &top1 = p.topK.front();
        LOG.info("Frame idx=%d Top1: %s (%d) %.2f%%",
                 p.idx,
                 top1.label.c_str(),
                 top1.classId,
                 top1.probability * 100.0f);
    }
}

} // namespace ResnetNodes
