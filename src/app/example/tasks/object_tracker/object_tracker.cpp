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

#include "object_tracker.h"
#include "custom_package.h"
namespace GryFlux
{
    std::shared_ptr<DataObject> ObjectTracker::process(const std::vector<std::shared_ptr<DataObject>> &inputs)
    {
        // 双输入
        if (inputs.size() != 2)
            return nullptr;

        auto detection = std::dynamic_pointer_cast<CustomPackage>(inputs[0]);
        auto feature_res = std::dynamic_pointer_cast<CustomPackage>(inputs[1]);

        // 默认输出为第一个输入，所有结果向第一个输入合并
        auto result = std::dynamic_pointer_cast<CustomPackage>(inputs[0]);
        // 添加处理逻辑
        std::vector<int> input_data;
        feature_res->get_data(input_data);

        for (auto i : input_data)
        {
            /* code */
            result->push_data(i);
        }
        

        return result;
    }
}
