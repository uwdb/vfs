#include "video.h"
#include "writablevideo.h"
#include "cuda_runtime.h"
#include "nppi.h"
#include "homography.h"
#include "projection.h"
#include "jointcompression.h"

#include <iostream>
#include <fuse/fuse.h>
#include "/home/bhaynes/projects/CudaSift/cudaSift.h"

namespace vfs {
    WritableVirtualVideo::WritableVirtualVideo(const VirtualVideo &base)
        : WritableVirtualVideo(base.name(), base.video(), base.format(), base.height(), base.width(), base.mode())
    { }


    WritableVirtualVideo::WritableVirtualVideo(const std::string &name, Video &video, VideoFormat format,
                                               size_t height, size_t width, mode_t mode)
        : VirtualVideo(name, video, format, height, width, mode),
          writer_(std::make_unique<graphics::JointWriter<3>>(height, width, graphics::OneTimeHomographyUpdate{})),
          buffer_(format.buffer_size(height, width)),
          head_(buffer_.begin()),
          tail_(head_)
          //written_(0u)
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
        //static std::array<char, 32*1024*1024> frame;
        const auto copy_size = std::min(size, static_cast<size_t>(std::distance(buffer_.end(), tail_)));

        if(tail_ < buffer_.end()) {
            std::copy(chunk, chunk + copy_size, tail_);
            std::advance(tail_, copy_size);
        }
        //if(current_ < buffer_.end()) {
        //    const auto copy_size = std::min(size, static_cast<size_t>(std::distance(buffer_.end(), current_)));
        //    std::copy(chunk, chunk + copy_size, current_ + copy_size);
        //    std::advance(current_, copy_size);
        //}
        //std::copy(chunk, chunk + size, buffer_.begin() + offset);

        //const auto height = 540u, width = 960u;
        //if(offset >= 1555200 - 4096) {
        //long d = std::distance(tail_, head_), f = format().frame_size(height, width).value();
        if(std::distance(head_, tail_) >= static_cast<ssize_t>(frame_size().value())) {
            //std::vector<unsigned char> left{buffer_.begin(), buffer_.begin() + offset + size};
            //std::vector<unsigned char> right{buffer_.begin(), buffer_.begin() + offset + size   };
            //std::vector<unsigned char> left{head_, head_ + frame_size().value()};
            //std::vector<unsigned char> right{head_, head_ + frame_size().value()};

            writer_->write(head_, head_);
            std::advance(head_, frame_size().value());
            if(head_ >= tail_)
                head_ = buffer_.begin();

            //graphics::GpuImage<3, Npp8u> gpu_left{left, nppiMalloc_8u_C3, height(), width()};
            //graphics::GpuImage<3, Npp8u> gpu_right{right, nppiMalloc_8u_C3, height(), width()};
            //graphics::SiftConfiguration configuration{height(), width()};

            //auto homography = graphics::find_homography(gpu_left, gpu_right, configuration);

            //graphics::PartitionBuffer partition{gpu_left, homography};
            //auto [pleft, overlap, pright] = graphics::partition(gpu_left, homography);
            //graphics::partition(gpu_left, partition);
            //graphics::GpuImage<3, Npp8u> result{nppiMalloc_8u_C3, overlap.height(), overlap.width()};
            //graphics::project(gpu_right, partition.overlap(), homography);

            //FILE *out = fopen("../input_left.rgb", "w");
            //auto data = gpu_left.download();
            //fwrite(data.data(), data.size(), sizeof(Npp8u), out);
            //fclose(out);

            //out = fopen("../projected.rgb", "w");
            //data = partition.overlap().download();
            //fwrite(data.data(), data.size(), sizeof(Npp8u), out);
            //fclose(out);

/*
            auto r2 = nppiWarpPerspective_8u_C3R(source, {width, height}, source_step, source_roi, target, target_step, target_roi, H, NPPI_INTER_NN);

            //auto r5 = nppiConvert_32f8u_C3R(fsource, fsource_step, target, 3 * target_step, convert_roi, NPP_ROUND_NEAREST_TIES_TO_EVEN);

            //auto r3 = cudaMemcpy2D(result.data(), sizeof(Npp8u) * 3 * width, source, sizeof(Npp8u) * source_step, sizeof(Npp8u) * 3 * width, height, cudaMemcpyDeviceToHost);
            auto r3 = cudaMemcpy2D(result.data(), sizeof(Npp8u) * 3 * width, target, sizeof(Npp8u) * target_step, sizeof(Npp8u) * 3 * width, height, cudaMemcpyDeviceToHost);
*/
        }

        return static_cast<int>(copy_size);
    }
} // namespace vfs
