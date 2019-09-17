#ifndef VFS_PROJECTION_H
#define VFS_PROJECTION_H

#include "homography.h"

namespace vfs::graphics {
    std::tuple<GpuImage<3, Npp8u>, GpuImage<3, Npp8u>, GpuImage<3, Npp8u>> partition(
            const GpuImage<3, Npp8u>&, const Homography&);
    void project(const GpuImage<3, Npp8u> &input, GpuImage<3, Npp8u> &output, const Homography&);
}

#endif //VFS_PROJECTION_H
