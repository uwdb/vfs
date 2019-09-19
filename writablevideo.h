#ifndef VFS_WRITABLEVIDEO_H
#define VFS_WRITABLEVIDEO_H

#include "inode.h"

namespace vfs {

    class WritableVirtualVideo: public VirtualVideo {
    public:
        WritableVirtualVideo(const VirtualVideo&);
        WritableVirtualVideo(const std::string&, Video&, VideoFormat, size_t, size_t, mode_t);

        int open(struct fuse_file_info&) override;
        int write(const char*, size_t, off_t, struct fuse_file_info&) override;
        int truncate(off_t) override { return 0; }

    private:
        std::vector<unsigned char> buffer_;
    };

} // namespace vfs

#endif //VFS_WRITABLEVIDEO_H
