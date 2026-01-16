/*************************************************************************************************************************
 * Copyright 2025 Sunhaihua1
 *
 * GryFlux Framework - Input Node
 *************************************************************************************************************************/
#pragma once

#include "framework/node_base.h"
#include "framework/data_packet.h"
#include "framework/context.h"

namespace PipelineNodes
{

/**
 * @brief Input node - Entry point of the pipeline
 *
 * This node marks the beginning of data packet processing.
 * Typically used for logging or initial validation.
 */
class InputNode : public GryFlux::NodeBase
{
public:
    /**
     * @brief Default constructor
     */
    InputNode() = default;

    /**
     * @brief Execute input node logic
     *
     * @param packet Data packet reference (borrow, not own)
     * @param ctx Context reference (NullContext for CPU tasks)
     */
    void execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx) override;
};

} // namespace PipelineNodes
