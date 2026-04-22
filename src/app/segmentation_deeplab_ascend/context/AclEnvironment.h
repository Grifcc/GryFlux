#pragma once

#include <string>

namespace YoloxRuntime {

class AclEnvironment {
public:
    static bool acquire(int device_id, std::string* error);
    static void release(int device_id);
};

}  // namespace YoloxRuntime
