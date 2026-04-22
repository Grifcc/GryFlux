#include "InputNode.h"

#include "packet/realesrgan_packet.h"
#include "utils/logger.h"

#include <filesystem>
#include <stdexcept>

namespace PipelineNodes
{

void InputNode::execute(GryFlux::DataPacket &packet, GryFlux::Context &ctx)
{
    (void)ctx;
    auto &p = static_cast<RealEsrganPacket &>(packet);

    if (p.lr_path.empty())
    {
        throw std::runtime_error("InputNode: lr_path is empty.");
    }

    const std::filesystem::path imagePath(p.lr_path);
    if (!std::filesystem::exists(imagePath))
    {
        throw std::runtime_error("InputNode: image file does not exist: " + p.lr_path);
    }

    p.lr_w = 0;
    p.lr_h = 0;
    p.lr_image.release();
    p.sr_image.release();
    p.output_buffer.clear();
    p.output_dims = aclmdlIODims{};
    p.output_format = ACL_FORMAT_UNDEFINED;
    p.output_data_type = ACL_DT_UNDEFINED;

    LOG.debug(
        "Packet %d: image=%s",
        p.frame_id,
        p.lr_path.c_str());
}

} // namespace PipelineNodes
