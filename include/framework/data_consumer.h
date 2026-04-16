/*************************************************************************************************************************
 * Copyright 2025 Grifcc & Sunhaihua1
 *
 * GryFlux Framework - Data Consumer Interface
 *************************************************************************************************************************/
#ifndef GRYFLUX_DATA_CONSUMER_H
#define GRYFLUX_DATA_CONSUMER_H

#include "data_packet.h"
#include <memory>

namespace GryFlux
{

/**
 * @brief 数据消费者接口 - 消费者
 *
 * 负责处理管道输出的结果数据包。
 * 用户可以实现此接口来支持不同的输出方式（文件、显示、网络等）。
 *
 * @example
 * @code
 * class VideoWriter : public DataConsumer {
 * public:
 *     void consume(std::unique_ptr<DataPacket> packet) override {
 *         auto& result = static_cast<ImagePacket&>(*packet);
 *         // Write result to video file
 *         videoFile.write(result.image);
 *     }
 * };
 * @endcode
 */
class DataConsumer
{
public:
    virtual ~DataConsumer() = default;

    /**
     * @brief 消费一个输出数据包
     *
     * 此方法会被 AsyncPipeline 调用，处理每个完成的数据包。
     *
     * @param packet 完成的数据包（unique_ptr，转移所有权）
     */
    virtual void consume(std::unique_ptr<DataPacket> packet) = 0;

    /**
     * @brief 消费失败数据包（可选）
     *
     * 当数据包在图执行中失败时，AsyncPipeline 会调用该接口。
     * 默认实现为 no-op；如需统一处理失败包，请在子类中重写。
     */
    virtual void consumeFailed(std::unique_ptr<DataPacket> packet)
    {
        (void)packet;
    }
};

} // namespace GryFlux

#endif // GRYFLUX_DATA_CONSUMER_H
