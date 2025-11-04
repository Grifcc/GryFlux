/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************************************************************/
#include "framework/node_profiler.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>

namespace GryFlux
{
    void NodeProfiler::printStats() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);

        if (stats_.empty())
        {
            std::cout << "No profiling data available." << std::endl;
            return;
        }

        // 按节点名称排序
        std::vector<std::pair<std::string, NodeStats>> sorted_stats(stats_.begin(), stats_.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(),
                  [](const auto &a, const auto &b)
                  { return a.first < b.first; });

        std::cout << "\n========================================" << std::endl;
        std::cout << "Node Performance Statistics" << std::endl;
        std::cout << "========================================" << std::endl;

        // 表头
        std::cout << std::left << std::setw(25) << "Node Name"
                  << std::right << std::setw(12) << "Count"
                  << std::setw(15) << "Avg (μs)"
                  << std::setw(15) << "Min (μs)"
                  << std::setw(15) << "Max (μs)"
                  << std::setw(15) << "Total (ms)"
                  << std::endl;

        std::cout << std::string(97, '-') << std::endl;

        // 数据行
        for (const auto &[name, stats] : sorted_stats)
        {
            uint64_t count = stats.count.load();
            uint64_t total_ns = stats.total_ns.load();
            uint64_t min_ns = stats.min_ns.load();
            uint64_t max_ns = stats.max_ns.load();
            uint64_t avg_ns = count > 0 ? total_ns / count : 0;

            // 转换为微秒和毫秒
            double avg_us = avg_ns / 1000.0;
            double min_us = min_ns / 1000.0;
            double max_us = max_ns / 1000.0;
            double total_ms = total_ns / 1000000.0;

            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(12) << count
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_us
                      << std::setw(15) << std::fixed << std::setprecision(2) << min_us
                      << std::setw(15) << std::fixed << std::setprecision(2) << max_us
                      << std::setw(15) << std::fixed << std::setprecision(2) << total_ms
                      << std::endl;
        }

        std::cout << "========================================\n"
                  << std::endl;
    }

} // namespace GryFlux
