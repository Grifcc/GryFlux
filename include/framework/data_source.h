/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
 *
 * GryFlux Framework - Data Source Interface
 *************************************************************************************************************************/
#ifndef GRYFLUX_DATA_SOURCE_H
#define GRYFLUX_DATA_SOURCE_H

#include "data_packet.h"
#include <memory>
#include <atomic>

namespace GryFlux
{

/**
 * @brief 数据源接口 - 生产者
 *
 * 负责持续产生数据包，提交到处理管道。
 * 用户可以实现此接口来支持不同的数据源（文件、摄像头、网络等）。
 *
 * @example
 * @code
 * class VideoSource : public DataSource {
 * public:
 *     std::unique_ptr<DataPacket> produce() override {
 *         auto packet = std::make_unique<ImagePacket>();
 *         // Read frame from video file
 *         // 读完后调用 setHasMore(false) 来停止生产
 *         return packet;
 *     }
 * };
 * @endcode
 */
class DataSource
{
public:
    virtual ~DataSource() = default;

    /**
     * @brief 主动请求数据源停止生产
     *
     * AsyncPipeline::stop() 会调用该接口，通知可能阻塞在 produce() 的数据源尽快退出。
     * 默认实现仅设置 hasMore=false；阻塞型数据源可重写该方法并执行唤醒动作。
     */
    virtual void requestStop()
    {
        setHasMore(false);
    }

    /**
     * @brief 产生一个数据包
     *
     * 此方法会被 AsyncPipeline 循环调用，每次调用产生一个新的数据包。
     *
     * @return 新创建的数据包（unique_ptr，转移所有权）
     */
    virtual std::unique_ptr<DataPacket> produce() = 0;

    /**
     * @brief 检查是否还有更多数据
     *
     * 这是一个非虚函数：框架统一通过内部状态变量判断是否继续生产。
     * 用户在 `produce()` 内部根据自身状态调用 `setHasMore(false)` 来停止生产。
     *
     * @return true  - 还有数据，继续调用 produce()
     *         false - 数据已耗尽，停止生产
     */
    bool hasMore() const
    {
        return hasMore_.load(std::memory_order_acquire);
    }

protected:
    void setHasMore(bool hasMore)
    {
        hasMore_.store(hasMore, std::memory_order_release);
    }

private:
    std::atomic<bool> hasMore_{true};
};

} // namespace GryFlux

#endif // GRYFLUX_DATA_SOURCE_H
