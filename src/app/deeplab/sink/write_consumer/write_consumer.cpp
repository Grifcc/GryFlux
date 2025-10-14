#include "write_consumer.h"
#include "package.h"

namespace GryFlux
{
    void WriteConsumer::run()
    {
        LOG.info("[WriteConsumer] Consumer started");

        // 只在生产者活跃时处理
        while (shouldContinue())
        {
            std::shared_ptr<DataObject> output;

            if (getData(output))
            {
                auto result = std::dynamic_pointer_cast<MaskPackage>(output);
                if (result)
                {
                    processedFrames++;
                    auto img = result->get_mask();
                    cv::imwrite( outputPath + "/output_" + std::to_string(processedFrames) + ".jpg", img);

                    LOG.info("Frame %d processed", processedFrames);
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        LOG.info("[TestConsumer] Processed frames: %d", processedFrames);
        LOG.info("[TestConsumer] Consumer finished");
    }
};
