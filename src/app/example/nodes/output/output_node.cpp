/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Object Tracker Node Implementation
 *************************************************************************************************************************/
#include "output_node.h"
#include "context/simulated_tracker_context.h"
#include "packet/simple_data_packet.h"
#include "utils/logger.h"

namespace PipelineNodes
{

void ObjectTrackerNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    auto &p = static_cast<SimpleDataPacket &>(packet);
    auto &tracker = static_cast<SimulatedTrackerContext &>(ctx);

    // ObjectTracker: trackVec[i] = detectionVec[i] + featureVec[i]
    // 融合两个并行分支的结果
    for (size_t i = 0; i < p.trackVec.size(); ++i)
    {
        p.trackVec[i] = p.detectionVec[i] + p.featureVec[i];
    }

    tracker.updateFrame(p.id);

    LOG.debug("Packet %d: ObjectTracker (tracker resource, size = %zu, lastFrame=%d)",
             p.id, p.trackVec.size(), tracker.getLastFrameId());
}

void FinalOutputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)packet;
    (void)ctx;
}

} // namespace PipelineNodes
