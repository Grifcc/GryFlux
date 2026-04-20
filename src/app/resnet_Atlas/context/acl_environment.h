#pragma once

#include <string>

namespace resnet {

class AclEnvironment {
public:
    static bool acquire(int device_id, std::string* error);
    static void release(int device_id);
};

}  // namespace resnet
