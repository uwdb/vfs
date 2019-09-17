#ifndef VFS_VIRTUALVIDEO_H
#define VFS_VIRTUALVIDEO_H

#include "inode.h"

namespace vfs {
    class VirtualVideo: public File {
    public:
        VirtualVideo(const std::string &name, Video&, size_t width, size_t height, mode_t);

        int open(struct fuse_file_info&) override;
        int truncate(off_t) override { return EACCES; }
        int read(const std::filesystem::path&, char*, size_t, off_t, struct fuse_file_info&) override;
        int write(const char*, size_t, off_t, struct fuse_file_info&) override { return EACCES; }

    protected:
        Video& video() const { return source_; } //source_.mount().find(source_.path()); }

    private:
        Video &source_;
        const size_t width_, height_;
    };

} // namespace vfs

#endif //VFS_VIRTUALVIDEO_H
