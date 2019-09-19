#ifndef VFS_PROJECTION_H
#define VFS_PROJECTION_H

#include "homography.h"

namespace vfs::graphics {
    class PartitionBuffer {
    public:
        PartitionBuffer(const GpuImage<3, Npp8u> &input, const Homography &homography)
            : homography_(homography),
              partitions_(homography.partitions(input.size())),
              widths_{partitions_.left.x0,
                      partitions_.left.x1 - partitions_.left.x0,
                      input.height() - partitions_.left.x0},
              frames_{make_frame(input.allocator(), input.height(), widths_.left),
                      make_frame(input.allocator(), input.height(), widths_.overlap),
                      make_frame(input.allocator(), input.height(), widths_.overlap)}
        { }

        const Partitions& partitions() const { return partitions_; }

        bool has_left() const { return frames_.left != nullptr; }
        bool has_overlap() const { return frames_.overlap != nullptr; }
        bool has_right() const { return frames_.right != nullptr; }
        GpuImage<3, Npp8u>& left() const { return *frames_.left; }
        GpuImage<3, Npp8u>& overlap() const { return *frames_.overlap; }
        GpuImage<3, Npp8u>& right() const { return *frames_.right; }

    private:
        std::unique_ptr<GpuImage<3, Npp8u>> make_frame(const GpuImage<3, Npp8u>::allocator_t &allocator,
                                                       const size_t height, const size_t width) {
            return height != 0 && width != 0
                ? std::make_unique<GpuImage<3, Npp8u>>(allocator, height, width)
                : nullptr;
        }

        Homography homography_;
        Partitions partitions_;
        struct {
            size_t left, overlap, right;
        } widths_;
        struct {
            std::unique_ptr<GpuImage<3, Npp8u>> left;
            std::unique_ptr<GpuImage<3, Npp8u>> overlap;
            std::unique_ptr<GpuImage<3, Npp8u>> right;
        } frames_;
    };

    PartitionBuffer& partition(const GpuImage<3, Npp8u> &input, PartitionBuffer &output);
    //std::tuple<GpuImage<3, Npp8u>, GpuImage<3, Npp8u>, GpuImage<3, Npp8u>> partition(
    //        const GpuImage<3, Npp8u>&, const Homography&);
    void project(const GpuImage<3, Npp8u> &input, GpuImage<3, Npp8u> &output, const Homography&);
}

#endif //VFS_PROJECTION_H
