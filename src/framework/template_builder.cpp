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
#include "framework/template_builder.h"
#include "utils/logger.h"
#include <stdexcept>

namespace GryFlux
{

    TemplateBuilder::TemplateBuilder(std::shared_ptr<GraphTemplate> tmpl)
        : template_(tmpl)
    {
    }

    void TemplateBuilder::setInputNode(const std::string &nodeId,
                                       std::shared_ptr<NodeBase> nodeImpl)
    {
        addTaskDefInternal(detail::NodeType::Input,
                           nodeId,
                           std::move(nodeImpl),
                           "",
                           {});
    }

    void TemplateBuilder::addTask(const std::string &nodeId,
                                  std::shared_ptr<NodeBase> nodeImpl,
                                  const std::string &resourceTypeName,
                                  const std::vector<std::string> &predecessorIds)
    {
        addTaskDefInternal(detail::NodeType::Task,
                           nodeId,
                           std::move(nodeImpl),
                           resourceTypeName,
                           predecessorIds);
    }

    void TemplateBuilder::setOutputNode(const std::string &nodeId,
                                        std::shared_ptr<NodeBase> nodeImpl,
                                        const std::vector<std::string> &predecessorIds)
    {
        addTaskDefInternal(detail::NodeType::Output,
                           nodeId,
                           std::move(nodeImpl),
                           "",
                           predecessorIds);
    }

    void TemplateBuilder::addTaskDefInternal(detail::NodeType type,
                                             const std::string &nodeId,
                                             std::shared_ptr<NodeBase> nodeImpl,
                                             const std::string &resourceTypeName,
                                             const std::vector<std::string> &predecessorIds)
    {
        if (!nodeImpl)
        {
            throw std::runtime_error("Node implementation is null for node '" + nodeId + "'");
        }

        if (nodeIdToIndex_.count(nodeId) > 0)
        {
            throw std::runtime_error("Node ID '" + nodeId + "' already exists");
        }

        if (type == detail::NodeType::Input)
        {
            if (!template_->tasks_.empty())
            {
                throw std::runtime_error("Input node must be added first (nodeId='" + nodeId + "')");
            }
            if (!predecessorIds.empty())
            {
                throw std::runtime_error("Input node cannot have predecessors (nodeId='" + nodeId + "')");
            }
        }

        detail::TaskDef node;
        node.type = type;
        node.nodeId = nodeId;
        node.resourceTypeName = (type == detail::NodeType::Task) ? resourceTypeName : "";
        node.nodeImpl = std::move(nodeImpl);

        // 解析前驱节点索引
        for (const auto &predId : predecessorIds)
        {
            auto it = nodeIdToIndex_.find(predId);
            if (it == nodeIdToIndex_.end())
            {
                throw std::runtime_error("Predecessor node '" + predId + "' not found for node '" + nodeId + "'");
            }
            node.parentIndices.push_back(it->second);
        }

        size_t index = template_->tasks_.size();
        template_->tasks_.push_back(std::move(node));
        nodeIdToIndex_[nodeId] = index;

        const char *typeStr = nullptr;
        switch (type)
        {
        case detail::NodeType::Input:
            typeStr = "input";
            break;
        case detail::NodeType::Task:
            typeStr = "task";
            break;
        case detail::NodeType::Output:
            typeStr = "output";
            break;
        }

        LOG.debug("Added %s node '%s' at index %zu with %zu predecessors",
                  typeStr ? typeStr : "unknown",
                  nodeId.c_str(),
                  index,
                  predecessorIds.size());
    }

    void TemplateBuilder::finalizeBuild()
    {
        // 建立反向链接（后继节点）
        for (size_t i = 0; i < template_->tasks_.size(); ++i)
        {
            auto &node = template_->tasks_[i];

            for (size_t predIdx : node.parentIndices)
            {
                // 给前驱节点添加当前节点为后继
                template_->tasks_[predIdx].childIndices.push_back(i);
            }
        }

        LOG.info("Finalized graph template with %zu nodes", template_->tasks_.size());
    }

} // namespace GryFlux
