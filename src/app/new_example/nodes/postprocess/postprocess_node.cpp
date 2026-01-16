/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Feature Extractor Node Implementation
 *************************************************************************************************************************/
#include "postprocess_node.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"
#include <chrono>
#include <thread>

namespace PipelineNodes
{

void FeatExtractorNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // FeatExtractor: featureVec[i] = preprocessedVec[i] + 5
    for (size_t i = 0; i < p.featureVec.size(); ++i)
    {
        p.featureVec[i] = p.preprocessedVec[i] + 5.0f;
    }

    LOG.debug("Packet %d: FeatExtractor (vec[i] + 5, size = %zu)",
             p.id, p.featureVec.size());

    // Simulate processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

} // namespace PipelineNodes
