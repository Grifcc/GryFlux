/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Profiling Gate
 *************************************************************************************************************************/
#pragma once

#include "framework/profiler/profiling_build_config.h"
#include "framework/profiler/graph_profiler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace GryFlux
{
    class DataPacket;

    namespace Profiling
    {
        using EventType = GraphProfiler::EventType;
        using Event = GraphProfiler::Event;

        template <bool Enabled>
        class NodeScopeT;

        template <>
        class NodeScopeT<true>
        {
        public:
            NodeScopeT(DataPacket *packet, const std::string &nodeId)
                : scope_(packet, nodeId)
            {
            }

            void markFailed() { scope_.markFailed(); }

        private:
            GraphProfiler::NodeExecutionScope scope_;
        };

        template <>
        class NodeScopeT<false>
        {
        public:
            NodeScopeT(DataPacket *, const std::string &) {}
            void markFailed() {}
        };

        using NodeScope = NodeScopeT<kBuildProfiling>;

        namespace detail
        {
            template <bool Enabled>
            struct AutoEnableT;

            template <>
            struct AutoEnableT<true>
            {
                AutoEnableT() { GraphProfiler::instance().setEnabled(true); }
            };

            template <>
            struct AutoEnableT<false>
            {
                constexpr AutoEnableT() noexcept = default;
            };

            inline AutoEnableT<kBuildProfiling> autoEnable{};
        } // namespace detail

        inline void setEnabled(bool enabled)
        {
            if constexpr (kBuildProfiling)
            {
                GraphProfiler::instance().setEnabled(enabled);
            }
            else
            {
                (void)enabled;
            }
        }

        inline bool isEnabled()
        {
            if constexpr (kBuildProfiling)
            {
                return GraphProfiler::instance().isEnabled();
            }
            return false;
        }

        inline void reset()
        {
            if constexpr (kBuildProfiling)
            {
                GraphProfiler::instance().reset();
            }
        }

        inline void recordNodeScheduled(DataPacket *packet, const std::string &nodeId)
        {
            if constexpr (kBuildProfiling)
            {
                GraphProfiler::instance().recordNodeScheduled(packet, nodeId);
            }
            else
            {
                (void)packet;
                (void)nodeId;
            }
        }

        inline void recordNodeFailed(DataPacket *packet, const std::string &nodeId, uint64_t durationNs)
        {
            if constexpr (kBuildProfiling)
            {
                GraphProfiler::instance().recordNodeFailed(packet, nodeId, durationNs);
            }
            else
            {
                (void)packet;
                (void)nodeId;
                (void)durationNs;
            }
        }

        inline std::vector<Event> snapshotEvents()
        {
            if constexpr (kBuildProfiling)
            {
                return GraphProfiler::instance().snapshotEvents();
            }
            return {};
        }

        inline void dumpTimelineJson(const std::string &filePath)
        {
            if constexpr (kBuildProfiling)
            {
                GraphProfiler::instance().dumpTimelineJson(filePath);
            }
            else
            {
                (void)filePath;
            }
        }
    } // namespace Profiling

} // namespace GryFlux
