#ifndef VFS_VIRTUALVIDEO_H
#define VFS_VIRTUALVIDEO_H

#include "inode.h"
#include "format.h"

namespace vfs {
    class VirtualVideo: public File {
    public:

        VirtualVideo(const std::string &name, Video&, VideoFormat format, size_t height, size_t width, mode_t);

        size_t height() const { return height_; }
        size_t width() const { return width_; }
        const VideoFormat &format() const { return format_; }
        Video& video() const { return source_; } //source_.mount().find(source_.path()); }

        int open(struct fuse_file_info&) override;
        int truncate(off_t) override { return EACCES; }
        int read(const std::filesystem::path&, char*, size_t, off_t, struct fuse_file_info&) override;
        int write(const char*, size_t, off_t, struct fuse_file_info&) override { return EACCES; }

    private:
        Video &source_;
        const VideoFormat format_;
        const size_t width_, height_;
    };

} // namespace vfs

#endif //VFS_VIRTUALVIDEO_H
