/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without limitation the
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
#include "framework/resource_pool.h"
#include "utils/logger.h"

namespace GryFlux
{

    void ResourcePool::registerResourceType(const std::string &typeName,
                                            std::vector<std::shared_ptr<Context>> instances,
                                            std::chrono::milliseconds acquireTimeout)
    {
        std::lock_guard<std::mutex> lock(resourceRegistryMutex_);

        if (instances.empty())
        {
            LOG.warning("Registering empty resource type: %s", typeName.c_str());
            return;
        }

        auto &pool = resourceTypePools_[typeName];
        if (!pool.waiters.empty())
        {
            LOG.warning("Re-registering resource type '%s' while %zu waiters pending; clearing waiters",
                        typeName.c_str(), pool.waiters.size());
            pool.waiters.clear();
        }
        pool.nextWaiterSeq = 0;
        pool.allContexts = std::move(instances);
        pool.acquireTimeout = acquireTimeout;

        // 初始化可用队列
        for (auto &ctx : pool.allContexts)
        {
            pool.availableContexts.push(ctx);
        }

        LOG.info("Registered resource type '%s' with %zu instances",
                 typeName.c_str(), pool.allContexts.size());
    }

    void ResourcePool::setAcquireTimeout(const std::string &typeName, std::chrono::milliseconds acquireTimeout)
    {
        std::lock_guard<std::mutex> lock(resourceRegistryMutex_);
        auto it = resourceTypePools_.find(typeName);
        if (it == resourceTypePools_.end())
        {
            LOG.error("Resource type '%s' not registered (setAcquireTimeout)", typeName.c_str());
            return;
        }
        it->second.acquireTimeout = acquireTimeout;
    }

    std::chrono::milliseconds ResourcePool::getAcquireTimeout(const std::string &typeName) const
    {
        std::lock_guard<std::mutex> lock(resourceRegistryMutex_);
        auto it = resourceTypePools_.find(typeName);
        if (it == resourceTypePools_.end())
        {
            return std::chrono::milliseconds(0);
        }
        return it->second.acquireTimeout;
    }

    std::shared_ptr<Context> ResourcePool::acquire(const std::string &typeName,
                                                    std::chrono::milliseconds timeout)
    {
        return acquire(typeName, timeout, nullptr);
    }

    std::shared_ptr<Context> ResourcePool::acquire(const std::string &typeName,
                                                   std::chrono::milliseconds timeout,
                                                   const std::atomic<bool> *cancelFlag)
    {
        return acquire(typeName, timeout, cancelFlag, 0);
    }

    std::shared_ptr<Context> ResourcePool::acquire(const std::string &typeName,
                                                   std::chrono::milliseconds timeout,
                                                   const std::atomic<bool> *cancelFlag,
                                                   Priority priority)
    {
        if (cancelFlag && cancelFlag->load(std::memory_order_acquire))
        {
            return nullptr;
        }

        std::unique_lock<std::mutex> registryLock(resourceRegistryMutex_);

        auto it = resourceTypePools_.find(typeName);
        if (it == resourceTypePools_.end())
        {
            LOG.error("Resource type '%s' not registered", typeName.c_str());
            return nullptr;
        }

        auto &pool = it->second;
        registryLock.unlock();

        std::unique_lock<std::mutex> poolLock(pool.poolMutex);

        if (!pool.availableContexts.empty())
        {
            auto context = pool.availableContexts.front();
            pool.availableContexts.pop();
            return context;
        }

        ResourceTypePool::Waiter waiter;
        const uint64_t seq = pool.nextWaiterSeq++;
        const auto key = ResourceTypePool::WaitKey{priority, seq};
        auto waiterIt = pool.waiters.emplace(key, &waiter);

        const auto deadline = (timeout == std::chrono::milliseconds::zero())
                                  ? std::chrono::steady_clock::time_point::max()
                                  : (std::chrono::steady_clock::now() + timeout);

        while (!waiter.satisfied)
        {
            if (cancelFlag && cancelFlag->load(std::memory_order_acquire))
            {
                pool.waiters.erase(waiterIt);
                return nullptr;
            }

            if (deadline == std::chrono::steady_clock::time_point::max())
            {
                waiter.cv.wait(poolLock);
            }
            else
            {
                if (waiter.cv.wait_until(poolLock, deadline) == std::cv_status::timeout)
                {
                    if (!waiter.satisfied)
                    {
                        pool.waiters.erase(waiterIt);
                        if (!(cancelFlag && cancelFlag->load(std::memory_order_acquire)))
                        {
                            LOG.warning("Timeout acquiring resource '%s'", typeName.c_str());
                        }
                        return nullptr;
                    }
                }
            }
        }

        // waiter.satisfied == true => assigned is valid
        return std::move(waiter.assigned);
    }

    void ResourcePool::release(const std::string &typeName, std::shared_ptr<Context> context)
    {
        if (!context)
        {
            LOG.warning("Attempting to release null context for resource '%s'", typeName.c_str());
            return;
        }

        std::unique_lock<std::mutex> registryLock(resourceRegistryMutex_);

        auto it = resourceTypePools_.find(typeName);
        if (it == resourceTypePools_.end())
        {
            LOG.error("Resource type '%s' not registered during release", typeName.c_str());
            return;
        }

        auto &pool = it->second;
        registryLock.unlock();

        {
            std::unique_lock<std::mutex> poolLock(pool.poolMutex);

            // Prefer waking the highest-priority waiter (smallest priority).
            if (!pool.waiters.empty())
            {
                auto first = pool.waiters.begin();
                auto *waiter = first->second;
                pool.waiters.erase(first);

                waiter->assigned = std::move(context);
                waiter->satisfied = true;

                poolLock.unlock();
                waiter->cv.notify_one();
                return;
            }

            pool.availableContexts.push(std::move(context));
        }
    }

    size_t ResourcePool::getAvailableCount(const std::string &typeName) const
    {
        std::lock_guard<std::mutex> registryLock(resourceRegistryMutex_);

        auto it = resourceTypePools_.find(typeName);
        if (it == resourceTypePools_.end())
        {
            return 0;
        }

        auto &pool = const_cast<ResourceTypePool &>(it->second);
        std::lock_guard<std::mutex> poolLock(pool.poolMutex);
        return pool.availableContexts.size();
    }

} // namespace GryFlux
