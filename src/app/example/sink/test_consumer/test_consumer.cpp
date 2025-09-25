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
#include "test_consumer.h"
#include "custom_package.h"

namespace GryFlux
{
    void TestConsumer::run()
    {
        LOG.info("[TestConsumer] Consumer started");

        // 只在生产者活跃时处理
        while (shouldContinue())
        {
            std::shared_ptr<DataObject> output;

            if (getData(output))
            {
                auto result = std::dynamic_pointer_cast<CustomPackage>(output);
                if (result)
                {
                    std::vector<int> data;
                    result->get_data(data);
                    for (auto &i : data)
                    {
                        LOG.info("Frame %d processed, data: %d", processedFrames, i);
                    }
                    processedFrames++;
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
