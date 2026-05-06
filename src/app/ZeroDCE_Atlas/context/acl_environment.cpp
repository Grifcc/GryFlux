#include "acl_environment.h"

#include "acl/acl.h"

#include <mutex>
#include <unordered_map>

namespace ZeroDce {
namespace {

std::mutex g_mutex;
size_t g_ref_count = 0;
std::unordered_map<int, size_t> g_device_ref_counts;

}  // namespace

bool AclEnvironment::acquire(int device_id, std::string* error) {
    std::lock_guard<std::mutex> lock(g_mutex);

    if (g_ref_count == 0) {
        const aclError init_ret = aclInit(nullptr);
        if (init_ret != ACL_ERROR_NONE) {
            if (error != nullptr) {
                *error = "aclInit failed, code=" + std::to_string(init_ret);
            }
            return false;
        }
    }

    auto device_it = g_device_ref_counts.find(device_id);
    if (device_it == g_device_ref_counts.end()) {
        const aclError set_ret = aclrtSetDevice(device_id);
        if (set_ret != ACL_ERROR_NONE) {
            if (g_ref_count == 0) {
                aclFinalize();
            }
            if (error != nullptr) {
                *error = "aclrtSetDevice failed, code=" + std::to_string(set_ret);
            }
            return false;
        }
        g_device_ref_counts.emplace(device_id, 1);
    } else {
        ++device_it->second;
    }

    ++g_ref_count;
    return true;
}

void AclEnvironment::release(int device_id) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_ref_count == 0) {
        return;
    }

    auto device_it = g_device_ref_counts.find(device_id);
    if (device_it == g_device_ref_counts.end()) {
        return;
    }

    if (device_it->second > 1) {
        --device_it->second;
    } else {
        g_device_ref_counts.erase(device_it);
        aclrtSetDevice(device_id);
        aclrtResetDevice(device_id);
    }

    --g_ref_count;
    if (g_ref_count == 0) {
        aclFinalize();
    }
}

}  // namespace ZeroDce
