/*************************************************************************************************************************
 * GryFlux Framework - Resource Scheduling & Precision Test
 *
 * 5 layers / 8 nodes DAG (includes input & output):
 *
 * Nodes: x, y, z, a, b, c, d, f
 * Dependencies (by formula):
 *   x = id
 *   y depends on x
 *   z depends on x
 *   a depends on y
 *   b depends on y
 *   c depends on y, z
 *   d depends on a, b, c
 *   f depends on d, z
 *
 * Per-packet formula (verifiable):
 *   x = id
 *   y = x + 1
 *   z = x * 2
 *   a = y + 10
 *   b = y * 3
 *   c = y + z
 *   d = a + b + c
 *   f = d + z = 9 * id + 15
 *************************************************************************************************************************/

#include "framework/async_pipeline.h"
#include "framework/graph_template.h"
#include "framework/profiler/profiling_build_config.h"
#include "framework/resource_pool.h"
#include "framework/template_builder.h"
#include "utils/logger.h"

#include "consumer/calc_consumer.h"
#include "context/adder_context.h"
#include "context/multiplier_context.h"
#include "nodes/add/add_nodes.h"
#include "nodes/input/input_node.h"
#include "nodes/mul/mul_nodes.h"
#include "nodes/output/output_node.h"
#include "source/calc_source.h"

#include <chrono>
#include <cstdlib>
#include <cstring>

namespace
{

void printUsage(const char *prog)
{
    LOG.info("Usage: %s [--num-packets N] [--mid-sleep-ms N] [--mid-sleep-s S] [--resource-timeout-ms N] [--adder-timeout-ms N] [--multiplier-timeout-ms N] [--help]", prog);
    LOG.info("  --num-packets N    Number of packets to run (default 200)");
    LOG.info("  --threads N        Thread pool size (default 12)");
    LOG.info("  --max-active N     Max active packets in flight (default 0 = auto)");
    LOG.info("  --serial           Force strictly serial run (equivalent to --threads 1 --max-active 1)");
    LOG.info("  --print-first N    Print result vs expected for first N packets (default 0)");
    LOG.info("  --print-all        Print result vs expected for all packets");
    LOG.info("  --mid-sleep-ms N   Extra sleep in milliseconds for intermediate nodes (y/z/a/b/c/d)");
    LOG.info("  --mid-sleep-s S    Extra sleep in seconds for intermediate nodes (double supported)");
    LOG.info("  --resource-timeout-ms N   Default resource acquire timeout in ms (default 0 = no timeout)");
    LOG.info("  --adder-timeout-ms N      Acquire timeout for resource type 'adder' in ms (default = --resource-timeout-ms)");
    LOG.info("  --multiplier-timeout-ms N Acquire timeout for resource type 'multiplier' in ms (default = --resource-timeout-ms)");
}

struct CliOptions
{
    size_t numPackets = 200;
    long long midSleepMs = 0;
    size_t threadPoolSize = 12;
    size_t maxActivePackets = 0;
    long long resourceTimeoutMs = 0;
    long long adderTimeoutMs = -1;
    long long multiplierTimeoutMs = -1;
    bool serial = false;
    size_t printFirstN = 0;
    bool printAll = false;
};

CliOptions parseOptions(int argc, char **argv)
{
    CliOptions opt;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
        {
            printUsage(argv[0]);
            std::exit(0);
        }

        if (std::strcmp(argv[i], "--mid-sleep-ms") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--mid-sleep-ms requires a value");
                std::exit(2);
            }
            opt.midSleepMs = std::atoll(argv[++i]);
            continue;
        }

        if (std::strcmp(argv[i], "--mid-sleep-s") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--mid-sleep-s requires a value");
                std::exit(2);
            }
            const double sec = std::atof(argv[++i]);
            opt.midSleepMs = static_cast<long long>(sec * 1000.0);
            continue;
        }

        if (std::strcmp(argv[i], "--resource-timeout-ms") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--resource-timeout-ms requires a value");
                std::exit(2);
            }
            opt.resourceTimeoutMs = std::atoll(argv[++i]);
            continue;
        }

        if (std::strcmp(argv[i], "--adder-timeout-ms") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--adder-timeout-ms requires a value");
                std::exit(2);
            }
            opt.adderTimeoutMs = std::atoll(argv[++i]);
            continue;
        }

        if (std::strcmp(argv[i], "--multiplier-timeout-ms") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--multiplier-timeout-ms requires a value");
                std::exit(2);
            }
            opt.multiplierTimeoutMs = std::atoll(argv[++i]);
            continue;
        }

        if (std::strcmp(argv[i], "--threads") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--threads requires a value");
                std::exit(2);
            }
            const long long v = std::atoll(argv[++i]);
            if (v <= 0)
            {
                LOG.error("--threads must be > 0");
                std::exit(2);
            }
            opt.threadPoolSize = static_cast<size_t>(v);
            continue;
        }

        if (std::strcmp(argv[i], "--max-active") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--max-active requires a value");
                std::exit(2);
            }
            const long long v = std::atoll(argv[++i]);
            if (v < 0)
            {
                LOG.error("--max-active must be >= 0");
                std::exit(2);
            }
            opt.maxActivePackets = static_cast<size_t>(v);
            continue;
        }

        if (std::strcmp(argv[i], "--serial") == 0)
        {
            opt.serial = true;
            continue;
        }

        if (std::strcmp(argv[i], "--print-all") == 0)
        {
            opt.printAll = true;
            continue;
        }

        if (std::strcmp(argv[i], "--print-first") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--print-first requires a value");
                std::exit(2);
            }
            const long long v = std::atoll(argv[++i]);
            if (v < 0)
            {
                LOG.error("--print-first must be >= 0");
                std::exit(2);
            }
            opt.printFirstN = static_cast<size_t>(v);
            continue;
        }

        if (std::strcmp(argv[i], "--num-packets") == 0)
        {
            if (i + 1 >= argc)
            {
                LOG.error("--num-packets requires a value");
                std::exit(2);
            }
            const long long v = std::atoll(argv[++i]);
            if (v <= 0)
            {
                LOG.error("--num-packets must be > 0");
                std::exit(2);
            }
            opt.numPackets = static_cast<size_t>(v);
            continue;
        }
    }

    if (opt.midSleepMs < 0)
    {
        opt.midSleepMs = 0;
    }

    if (opt.resourceTimeoutMs < 0)
    {
        opt.resourceTimeoutMs = 0;
    }

    if (opt.adderTimeoutMs < -1)
    {
        opt.adderTimeoutMs = -1;
    }

    if (opt.multiplierTimeoutMs < -1)
    {
        opt.multiplierTimeoutMs = -1;
    }

    if (opt.serial)
    {
        opt.threadPoolSize = 1;
        opt.maxActivePackets = 1;
    }

    // Convenience: if user asks to print all, no need to also pass a huge N.
    if (opt.printAll)
    {
        opt.printFirstN = 0;
    }

    return opt;
}

} // namespace

int main(int argc, char** argv)
{
    LOG.setLevel(GryFlux::LogLevel::INFO);
    LOG.setOutputType(GryFlux::LogOutputType::CONSOLE);
    LOG.setAppName("ParallelCalcTest");

    const auto opt = parseOptions(argc, argv);
    const long long midSleepMs = opt.midSleepMs;

    LOG.info("========================================");
    LOG.info("  GryFlux Resource Scheduling Test");
    LOG.info("  DAG: 5 layers / 8 nodes");
    LOG.info("========================================");
    LOG.info("Extra mid-node sleep: %lld ms", midSleepMs);
    LOG.info("Thread pool size: %zu", opt.threadPoolSize);
    LOG.info("Max active packets: %zu", opt.maxActivePackets);
    LOG.info("Resource acquire timeout: %lld ms", opt.resourceTimeoutMs);
    const long long adderTimeoutMs = (opt.adderTimeoutMs >= 0) ? opt.adderTimeoutMs : opt.resourceTimeoutMs;
    const long long multiplierTimeoutMs = (opt.multiplierTimeoutMs >= 0) ? opt.multiplierTimeoutMs : opt.resourceTimeoutMs;
    LOG.info("Resource acquire timeout (adder): %lld ms", adderTimeoutMs);
    LOG.info("Resource acquire timeout (multiplier): %lld ms", multiplierTimeoutMs);
    LOG.info("Serial mode: %s", opt.serial ? "true" : "false");
    LOG.info("Print first N packets: %zu", opt.printFirstN);
    LOG.info("Print all packets: %s", opt.printAll ? "true" : "false");

    // 1) Resource pool
    auto resourcePool = std::make_shared<GryFlux::ResourcePool>();

    // Make multiplier more scarce to stress scheduling
    resourcePool->registerResourceType(
        "adder",
        {std::make_shared<AdderContext>(0), std::make_shared<AdderContext>(1)},
        std::chrono::milliseconds(adderTimeoutMs));
    resourcePool->registerResourceType(
        "multiplier",
        {std::make_shared<MultiplierContext>(0)},
        std::chrono::milliseconds(multiplierTimeoutMs));

    LOG.info("Registered resources: adder=2, multiplier=1");

    // 2) Graph template
    auto graphTemplate = GryFlux::GraphTemplate::buildOnce(
        [midSleepMs](GryFlux::TemplateBuilder* builder)
        {
            // Node x (input)
            builder->setInputNode<TestNodes::InputNode>("x");

            // Layer 2: y, z
            builder->addTask<TestNodes::AddConstNode>(
                "y",
                "adder",
                {"x"},
                &CalcPacket::x,
                &CalcPacket::y,
                1.0,
                midSleepMs);

            builder->addTask<TestNodes::MulConstNode>(
                "z",
                "multiplier",
                {"x"},
                &CalcPacket::x,
                &CalcPacket::z,
                2.0,
                midSleepMs);

            // Layer 3: a, b, c
            builder->addTask<TestNodes::AddConstNode>(
                "a",
                "adder",
                {"y"},
                &CalcPacket::y,
                &CalcPacket::a,
                10.0,
                midSleepMs);

            builder->addTask<TestNodes::MulConstNode>(
                "b",
                "multiplier",
                {"y"},
                &CalcPacket::y,
                &CalcPacket::b,
                3.0,
                midSleepMs);

            builder->addTask<TestNodes::Add2Node>(
                "c",
                "adder",
                {"y", "z"},
                &CalcPacket::y,
                &CalcPacket::z,
                &CalcPacket::c,
                midSleepMs);

            // Layer 4: d
            builder->addTask<TestNodes::Fuse3SumNode>(
                "d",
                "adder",
                {"a", "b", "c"},
                &CalcPacket::a,
                &CalcPacket::b,
                &CalcPacket::c,
                &CalcPacket::d,
                midSleepMs);

            // Layer 5: f (output)
            builder->setOutputNode<TestNodes::OutputNode>("f", {"d", "z"});
        });

    // 3) Source / consumer
    const size_t numPackets = opt.numPackets;
    auto source = std::make_shared<CalcSource>(numPackets);
    auto consumer = std::make_shared<CalcConsumer>(opt.printFirstN, opt.printAll);

    // 4) Pipeline
    GryFlux::AsyncPipeline pipeline(
        source,
        graphTemplate,
        resourcePool,
        consumer,
        opt.threadPoolSize,
        opt.maxActivePackets);


    if constexpr (GryFlux::Profiling::kBuildProfiling)
    {
        pipeline.setProfilingEnabled(true);
    }

    auto t0 = std::chrono::steady_clock::now();
    pipeline.run();
    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    LOG.info("========================================");
    LOG.info("All %zu packets completed in %lld ms", numPackets, static_cast<long long>(ms));
    LOG.info("Average: %.3f ms/packet", static_cast<double>(ms) / static_cast<double>(numPackets));
    LOG.info("Throughput: %.2f packets/sec", 1000.0 * static_cast<double>(numPackets) / static_cast<double>(ms));
    LOG.info("Verification: pass=%zu fail=%zu", consumer->getSuccessCount(), consumer->getFailureCount());
    LOG.info("========================================");

    if constexpr (GryFlux::Profiling::kBuildProfiling)
    {
        pipeline.printProfilingStats();
        pipeline.dumpProfilingTimeline("parallel_calc_timeline.json");
        LOG.info("Timeline dumped to parallel_calc_timeline.json");
    }

    return (consumer->getFailureCount() == 0) ? 0 : 1;
}
