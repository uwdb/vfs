#include "mount.h"
#include "video.h"

namespace vfs {

    Video::Video(const std::string &name, const Mount &mount,
                 mode_t mode) //, std::vector<std::filesystem::path> sources)
            : Directory(name, mount.root(), mode)  //, sources_(std::move(sources))
    { }

}