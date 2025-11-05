/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Graph Profiler
 *************************************************************************************************************************/
#pragma once

#include "framework/data_packet.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace GryFlux
{
    class GraphProfiler
    {
    public:
        enum class EventType
        {
            Scheduled,
            Started,
            Finished,
            Failed
        };

        struct Event
        {
            EventType type;
            uint64_t timestampNs;
            std::string nodeId;
            uint64_t packetId;
            std::string threadId;
            uint64_t durationNs;
        };

        static GraphProfiler &instance();

        void setEnabled(bool enabled);
        bool isEnabled() const;

        void reset();

        void recordNodeScheduled(DataPacket *packet, const std::string &nodeId);
        void recordNodeStarted(DataPacket *packet, const std::string &nodeId);
        void recordNodeFinished(DataPacket *packet, const std::string &nodeId, uint64_t durationNs);
        void recordNodeFailed(DataPacket *packet, const std::string &nodeId, uint64_t durationNs);

        std::vector<Event> snapshotEvents() const;
        void dumpTimelineJson(const std::string &filePath) const;

        class NodeExecutionScope
        {
        public:
            NodeExecutionScope(DataPacket *packet, const std::string &nodeId);
            ~NodeExecutionScope();

            void markFailed();

        private:
            GraphProfiler &profiler_;
            DataPacket *packet_;
            std::string nodeId_;
            std::chrono::steady_clock::time_point start_{};
            bool enabled_;
            bool failed_;
        };

    private:
        GraphProfiler();

        uint64_t nowSystemNs() const;
        std::string threadIdString() const;

        mutable std::mutex mutex_;
        std::vector<Event> events_;
        std::atomic<bool> enabled_;
    };

} // namespace GryFlux
