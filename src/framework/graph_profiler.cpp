/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Graph Profiler Implementation
 *************************************************************************************************************************/
#include "framework/profiler/graph_profiler.h"
#include "utils/logger.h"
#include <fstream>
#include <iomanip>
#include <sstream>

namespace GryFlux
{
    GraphProfiler &GraphProfiler::instance()
    {
        static GraphProfiler profiler;
        return profiler;
    }

    GraphProfiler::GraphProfiler()
        : enabled_(false)
    {
    }

    void GraphProfiler::setEnabled(bool enabled)
    {
        enabled_.store(enabled, std::memory_order_relaxed);
    }

    bool GraphProfiler::isEnabled() const
    {
        return enabled_.load(std::memory_order_relaxed);
    }

    void GraphProfiler::reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        events_.clear();
    }

    void GraphProfiler::recordNodeScheduled(DataPacket *packet, const std::string &nodeId)
    {
        if (!isEnabled())
        {
            return;
        }

        uint64_t packetId = packet ? packet->getId() : 0;

        Event event{
            EventType::Scheduled,
            nowSystemNs(),
            nodeId,
            packetId,
            threadIdString(),
            0};

        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }

    void GraphProfiler::recordNodeStarted(DataPacket *packet, const std::string &nodeId)
    {
        if (!isEnabled())
        {
            return;
        }

        uint64_t packetId = packet ? packet->getId() : 0;

        Event event{
            EventType::Started,
            nowSystemNs(),
            nodeId,
            packetId,
            threadIdString(),
            0};

        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }

    void GraphProfiler::recordNodeFinished(DataPacket *packet, const std::string &nodeId, uint64_t durationNs)
    {
        if (!isEnabled())
        {
            return;
        }

        uint64_t packetId = packet ? packet->getId() : 0;

        Event event{
            EventType::Finished,
            nowSystemNs(),
            nodeId,
            packetId,
            threadIdString(),
            durationNs};

        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }

    void GraphProfiler::recordNodeFailed(DataPacket *packet, const std::string &nodeId, uint64_t durationNs)
    {
        if (!isEnabled())
        {
            return;
        }

        uint64_t packetId = packet ? packet->getId() : 0;

        Event event{
            EventType::Failed,
            nowSystemNs(),
            nodeId,
            packetId,
            threadIdString(),
            durationNs};

        std::lock_guard<std::mutex> lock(mutex_);
        events_.push_back(std::move(event));
    }

    std::vector<GraphProfiler::Event> GraphProfiler::snapshotEvents() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return events_;
    }

    void GraphProfiler::dumpTimelineJson(const std::string &filePath) const
    {
        std::vector<Event> eventsCopy = snapshotEvents();

        std::ofstream ofs(filePath);
        if (!ofs)
        {
            LOG.error("Failed to write profiling timeline to %s", filePath.c_str());
            return;
        }

        ofs << "{\n  \"events\": [\n";
        for (size_t i = 0; i < eventsCopy.size(); ++i)
        {
            const auto &evt = eventsCopy[i];
            ofs << "    {";
            ofs << "\"type\": \"" << ([&evt]() {
                switch (evt.type)
                {
                case EventType::Scheduled:
                    return "scheduled";
                case EventType::Started:
                    return "started";
                case EventType::Finished:
                    return "finished";
                case EventType::Failed:
                    return "failed";
                }
                return "unknown";
            })()
                << "\", ";
            ofs << "\"timestamp_ns\": " << evt.timestampNs << ", ";
            ofs << "\"node\": \"" << evt.nodeId << "\", ";
            ofs << "\"packet_id\": " << evt.packetId << ", ";
            ofs << "\"thread\": \"" << evt.threadId << "\", ";
            ofs << "\"duration_ns\": " << evt.durationNs;
            ofs << "}";
            if (i + 1 < eventsCopy.size())
            {
                ofs << ",";
            }
            ofs << "\n";
        }
        ofs << "  ]\n}\n";
    }

    uint64_t GraphProfiler::nowSystemNs() const
    {
        auto now = std::chrono::system_clock::now().time_since_epoch();
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
    }

    std::string GraphProfiler::threadIdString() const
    {
        std::ostringstream oss;
        oss << std::this_thread::get_id();
        return oss.str();
    }

    GraphProfiler::NodeExecutionScope::NodeExecutionScope(DataPacket *packet, const std::string &nodeId)
        : profiler_(GraphProfiler::instance()),
          packet_(packet),
          nodeId_(nodeId),
          enabled_(profiler_.isEnabled()),
          failed_(false)
    {
        if (enabled_)
        {
            profiler_.recordNodeStarted(packet_, nodeId_);
            start_ = std::chrono::steady_clock::now();
        }
    }

    GraphProfiler::NodeExecutionScope::~NodeExecutionScope()
    {
        if (!enabled_)
        {
            return;
        }

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();

        if (failed_)
        {
            profiler_.recordNodeFailed(packet_, nodeId_, static_cast<uint64_t>(duration));
        }
        else
        {
            profiler_.recordNodeFinished(packet_, nodeId_, static_cast<uint64_t>(duration));
        }
    }

    void GraphProfiler::NodeExecutionScope::markFailed()
    {
        failed_ = true;
    }

} // namespace GryFlux
