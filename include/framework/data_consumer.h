/*************************************************************************************************************************
 * Copyright 2025 Grifcc
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
};

} // namespace GryFlux

#endif // GRYFLUX_DATA_CONSUMER_H
