/*************************************************************************************************************************
 * Copyright 2025 Grifcc
 *
 * GryFlux Framework - Data Source Interface
 *************************************************************************************************************************/
#ifndef GRYFLUX_DATA_SOURCE_H
#define GRYFLUX_DATA_SOURCE_H

#include "data_packet.h"
#include <memory>

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
 *         return packet;
 *     }
 *
 *     bool hasMore() override {
 *         return !videoFile.eof();
 *     }
 * };
 * @endcode
 */
class DataSource
{
public:
    virtual ~DataSource() = default;

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
     * @return true  - 还有数据，继续调用 produce()
     *         false - 数据已耗尽，停止生产
     */
    virtual bool hasMore() = 0;
};

} // namespace GryFlux

#endif // GRYFLUX_DATA_SOURCE_H
