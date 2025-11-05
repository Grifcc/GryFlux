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
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace GryFlux
{

    // 前向声明
    class GraphTemplate;

    /**
     * @brief 数据包基类 - 携带用户数据和执行状态
     *
     * 用户应继承此类，定义自己的数据结构，并预分配所有中间结果空间。
     *
     * 示例：
     * struct VideoFramePacket : public DataPacket {
     *     cv::Mat rawFrame;                   // 输入
     *     cv::Mat preprocessedImage;          // 中间结果（预分配）
     *     std::vector<Detection> detections;  // 中间结果（预分配）
     *
     *     VideoFramePacket() {
     *         preprocessedImage = cv::Mat(640, 640, CV_8UC3);
     *         detections.reserve(100);
     *     }
     * };
     */
    class DataPacket
    {
    public:
        DataPacket() = default;
        virtual ~DataPacket() = default;

        // 禁止拷贝（防止意外复制大对象和执行状态）
        DataPacket(const DataPacket &) = delete;
        DataPacket &operator=(const DataPacket &) = delete;

        // 禁止移动（执行状态包含原子变量，不可移动）
        DataPacket(DataPacket &&) = delete;
        DataPacket &operator=(DataPacket &&) = delete;

        /**
         * @brief 内嵌的执行状态
         */
        struct ExecutionState
        {
            std::shared_ptr<GraphTemplate> graphTemplate;

            // 每个节点的运行时状态
            struct NodeState
            {
                std::atomic<size_t> unfinishedPredecessorCount{0};
                std::atomic<bool> hasBeenEnqueued{false};
                std::atomic<bool> isCompleted{false};
            };

            std::vector<std::unique_ptr<NodeState>> nodeStates;
            std::atomic<bool> isGraphCompleted{false};
        };

        ExecutionState executionState_;

        /**
         * @brief 返回数据包的唯一 ID（必须实现）
         */
        virtual uint64_t getId() const = 0;

        /**
         * @brief 初始化执行状态（框架调用）
         *
         * @param tmpl 图模板
         */
        void initializeExecution(std::shared_ptr<GraphTemplate> tmpl);

        /**
         * @brief 尝试标记节点为就绪（原子操作，防止重复调度）
         *
         * @param nodeIndex 节点索引
         * @return true 如果成功标记为就绪，应该调度此节点
         */
        bool tryMarkNodeReady(size_t nodeIndex);

        /**
         * @brief 通知某个节点：它的一个前驱完成了
         *
         * @param nodeIndex 节点索引
         */
        void notifyPredecessorCompleted(size_t nodeIndex);

        /**
         * @brief 标记节点完成
         *
         * @param nodeIndex 节点索引
         */
        void markNodeCompleted(size_t nodeIndex);

        /**
         * @brief 检查整个图是否完成
         *
         * @return true 如果图执行完成
         */
        bool isCompleted() const;
    };

} // namespace GryFlux
