#include <fuse.h>
#include "inode.h"
#include "video.h"
#include "virtualvideo.h"

namespace vfs {

VirtualVideo::VirtualVideo(const std::string &name, Video& source, VideoFormat format, size_t height, size_t width,
                           size_t framerate, size_t gop_size, mode_t mode)
        : File(name, source, mode), source_(source), format_(format),
          width_(width), height_(height), framerate_(framerate), gop_size_(gop_size),
          frame_size_(format.frame_size(height, width))
{ }

int VirtualVideo::open(struct fuse_file_info &info) {
    if(info.flags & (O_RDWR | O_APPEND))
        return -EACCES;
    else if(info.flags & O_WRONLY) {
        //std::unique_ptr<Inode> writable = std::make_unique<WritableVirtualVideo>();
        auto &writable = video().substitute(*this, std::make_unique<WritableVirtualVideo>(
                name(), video(), format(), height(), width(), framerate(), gop_size(), mode()));
        return writable.open(info); //video().write(*this, info);
    } else {
        File::open(info);
        video().opened(*this);

        info.fh = (uintptr_t)this;
        return 0;
    }
}


int VirtualVideo::read(const std::filesystem::path &path, char *buffer, size_t size, off_t offset, struct fuse_file_info &info) {
if(offset < 5) {
memcpy(buffer, "A", 2);
return 1;
}
return 0;
}

//int VirtualVideo::write(const char*, size_t, off_t, struct fuse_file_info&) {
//return 0;
//}

}
