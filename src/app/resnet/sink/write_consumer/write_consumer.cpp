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
                // 处理图像结果
                auto image_result = std::dynamic_pointer_cast<ImagePackage>(output);
                if (image_result)
                {
                    processedFrames++;
                    auto img = image_result->get_data();
                    cv::imwrite(outputPath + "/output_" + std::to_string(processedFrames) + ".jpg", img);

                    LOG.info("Frame %d processed", processedFrames);
                }
            }
            else
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        LOG.info("[WriteConsumer] Processed frames: %d", processedFrames);
        LOG.info("[WriteConsumer] Consumer finished");
    }
};
