#include "video.h"
#include "writablevideo.h"
#include "cuda_runtime.h"
#include "nppi.h"
#include "homography.h"
#include "projection.h"
#include "jointcompression.h"
#include "compressionwriter.h"

#include <iostream>
#include <fuse/fuse.h>
#include "/home/bhaynes/projects/CudaSift/cudaSift.h"

#define HACK_INTERLEAVE_MULTIPLIER 2

namespace vfs {
    std::unique_ptr<VideoWriter> WritableVirtualVideo::get_writer(Video &video, const size_t height, const size_t width) {
        return std::make_unique<JointWriter<3>>(height, width, OneTimeHomographyUpdate{});
    }

    WritableVirtualVideo::WritableVirtualVideo(const VirtualVideo &base)
        : WritableVirtualVideo(base.name(), base.video(), base.format(), base.height(), base.width(), base.mode())
    { }

    WritableVirtualVideo::WritableVirtualVideo(const std::string &name, Video &video, VideoFormat format,
                                               size_t height, size_t width, mode_t mode)
        : VirtualVideo(name, video, format, height, width, mode),
          writer_(get_writer(video, height, width)),
          buffer_(/*TODO*/HACK_INTERLEAVE_MULTIPLIER * format.buffer_size(height, width)),
          head_(buffer_.begin()),
          tail_(head_),
          written_(0u)
    { }

    int WritableVirtualVideo::open(struct fuse_file_info &info) {
        video().opened(*this);
        info.nonseekable = 1;
        info.fh = (uintptr_t)this;
        /*if(info.flags & (O_RDWR | O_APPEND | O_WRONLY))
            return -EACCES;
        else if(info.flags & O_WRONLY)
            return -EACCES;
            //return video().write(info);
        else {
            File::open(info);

            info.fh = (uintptr_t)this;
            return 0;
        }*/
        return 0;
    }

    int WritableVirtualVideo::write(const char* chunk, size_t size, off_t, struct fuse_file_info&) {
        const auto write_size = std::max(0l, std::min(static_cast<ssize_t>(size), static_cast<ssize_t>(std::distance(tail_, buffer_.end()))));

        if(write_size > 0) {
        //if(tail_  < buffer_.end()) {
            std::copy(chunk, chunk + write_size, tail_);
            std::advance(tail_, write_size);
            written_ += write_size;
        }

        while(std::distance(head_, tail_) >= HACK_INTERLEAVE_MULTIPLIER * static_cast<ssize_t>(frame_size().value())) {
            const auto &left = head_;
            const auto &right = head_ + frame_size().value();

            writer_->write(left, right);
            std::advance(head_, HACK_INTERLEAVE_MULTIPLIER * frame_size().value());
            if(head_ >= tail_)
                head_ = tail_ = buffer_.begin();
        }

        return static_cast<int>(write_size);
    }

    int WritableVirtualVideo::flush(struct fuse_file_info &info) {
        writer_->flush();
        return 0;
    }

} // namespace vfs
