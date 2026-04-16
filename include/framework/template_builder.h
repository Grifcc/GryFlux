/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
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

#include "graph_template.h"
#include "node_base.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <type_traits>

namespace GryFlux
{

    /**
     * @brief 图构建器 - 用户友好的图构建接口
     *
     * 用于构建DAG图的拓扑结构。
     */
    class TemplateBuilder
    {
    public:
        explicit TemplateBuilder(std::shared_ptr<GraphTemplate> tmpl);

        /**
         * @brief 设置输入节点
         *
         * @param nodeId 节点ID
         * @param nodeImpl 节点实现对象（必须继承 NodeBase）
         */
        void setInputNode(const std::string &nodeId,
                          std::shared_ptr<NodeBase> nodeImpl);

        /**
         * @brief 设置输入节点（模板版本）- 简化类节点创建
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param args 转发给 NodeType 构造函数的参数
         */
        template<typename NodeType, typename... Args>
        void setInputNode(const std::string &nodeId, Args&&... args)
        {
            static_assert(std::is_base_of<NodeBase, NodeType>::value,
                          "NodeType must derive from GryFlux::NodeBase");
            setInputNode(nodeId, std::make_shared<NodeType>(std::forward<Args>(args)...));
        }

        /**
         * @brief 添加任务节点
         *
         * @param nodeId 节点ID
         * @param nodeImpl 节点实现对象（必须继承 NodeBase）
         * @param resourceTypeName 资源类型名称（空字符串表示CPU任务）
         * @param predecessorIds 前驱节点ID列表
         */
        void addTask(const std::string &nodeId,
                     std::shared_ptr<NodeBase> nodeImpl,
                     const std::string &resourceTypeName,
                     const std::vector<std::string> &predecessorIds);

        /**
         * @brief 添加类节点（模板版本）- 简化类节点创建
         *
         * 用户友好的接口，自动创建节点对象并管理生命周期。
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param resourceTypeName 资源类型名称（空字符串表示CPU任务）
         * @param predecessorIds 前驱节点ID列表
         * @param args 转发给 NodeType 构造函数的参数
         *
         * @example
         * @code
         * // 创建带配置参数的后处理节点
         * builder->addTask<PostprocessNode>(
         *     "postprocess",
         *     "",              // CPU task
         *     {"inference"},   // Dependencies
         *     0.45f,           // NMS threshold
         *     0.5f             // Confidence threshold
         * );
         * @endcode
         */
        template<typename NodeType, typename... Args>
        void addTask(const std::string &nodeId,
                     const std::string &resourceTypeName,
                     const std::vector<std::string> &predecessorIds,
                     Args&&... args)
        {
            static_assert(std::is_base_of<NodeBase, NodeType>::value,
                          "NodeType must derive from GryFlux::NodeBase");
            addTask(nodeId,
                    std::make_shared<NodeType>(std::forward<Args>(args)...),
                    resourceTypeName,
                    predecessorIds);
        }

        /**
         * @brief 设置输出节点
         *
         * @param nodeId 节点ID
         * @param nodeImpl 节点实现对象（必须继承 NodeBase）
         * @param predecessorIds 前驱节点ID列表
         */
        void setOutputNode(const std::string &nodeId,
                           std::shared_ptr<NodeBase> nodeImpl,
                           const std::vector<std::string> &predecessorIds);

        /**
         * @brief 设置输出节点（模板版本）- 简化类节点创建
         *
         * @tparam NodeType 节点类类型（必须继承 NodeBase）
         * @tparam Args 构造参数类型（自动推导）
         *
         * @param nodeId 节点ID
         * @param predecessorIds 前驱节点ID列表
         * @param args 转发给 NodeType 构造函数的参数
         */
        template<typename NodeType, typename... Args>
        void setOutputNode(const std::string &nodeId,
                           const std::vector<std::string> &predecessorIds,
                           Args&&... args)
        {
            static_assert(std::is_base_of<NodeBase, NodeType>::value,
                          "NodeType must derive from GryFlux::NodeBase");
            setOutputNode(nodeId,
                          std::make_shared<NodeType>(std::forward<Args>(args)...),
                          predecessorIds);
        }

        /**
         * @brief 完成构建（建立后继链接）
         */
        void finalizeBuild();

    private:
        void addTaskDefInternal(detail::NodeType type,
                                const std::string &nodeId,
                                std::shared_ptr<NodeBase> nodeImpl,
                                const std::string &resourceTypeName,
                                const std::vector<std::string> &predecessorIds);

        std::shared_ptr<GraphTemplate> template_;
        std::unordered_map<std::string, size_t> nodeIdToIndex_;
        bool hasOutputNode_ = false;
    };

} // namespace GryFlux
