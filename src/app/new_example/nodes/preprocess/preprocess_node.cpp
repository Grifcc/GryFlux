/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Image Preprocess Node Implementation
 *************************************************************************************************************************/
#include "preprocess_node.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"
#include <chrono>
#include <thread>

namespace PipelineNodes
{

void ImagePreprocessNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);

    // ImagePreprocess: preprocessedVec[i] = rawVec[i] * 2
    for (size_t i = 0; i < p.preprocessedVec.size(); ++i)
    {
        p.preprocessedVec[i] = p.rawVec[i] * 2.0f;
    }

    LOG.debug("Packet %d: ImagePreprocess (vec[i] * 2, size = %zu)",
             p.id, p.preprocessedVec.size());

    // Simulate processing time
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

} // namespace PipelineNodes
