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

#include "object_detector.h"
#include "custom_package.h"
#include <thread>
namespace GryFlux
{
    std::shared_ptr<DataObject> ObjectDetector::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        if (inputs.empty())
            return nullptr;

        auto input_data = std::dynamic_pointer_cast<CustomPackage>(inputs[0]);

        std::vector<int> data;
        input_data->get_data(data);

        // 在原有input数据的基础上添加1000-2000
        auto result = std::make_shared<CustomPackage>();
        for (auto i : data) result->push_data(i);
        for (int i = 0; i < 1000; i++)
        {
            /* code */
            result->push_data(i+1000);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return result;
    }
}
