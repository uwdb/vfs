#include "projection.h"

#include <chrono>

namespace vfs::graphics {

/*
def partition(H, Hi, left):
    # Left overlap at x0
    i = np.array([0, 0, 1])
    #Hi = np.linalg.inv(H)
    H0 = Hi.dot(i)
    H0n = H0 / H0[2]
    x0 = H0n[0]
    p0 = int(x0)

    p0 = int(Hi[0, 2] / Hi[2, 2])

    # Right of overlap at x1
    right = left.shape[1] * 3/4
    j = np.array([right, 0, 1])
    H1 = H.dot(j)
    H1n = H1 / H1[2]
    x1 = H1n[0]
    p1 = int(x1)

    ymin = int(H0n[1]) / 2 # Not sure about this
    #print H0n
    #Hy = Hi.dot(np.array([left.shape[1]*3/4, left.shape[0], 1]))
    #Hyn = Hy / Hy[2]
    ymax = left.shape[0]-ymin  # used symmetry, but should calculate explicitly
    #print ymin, ymax, left.shape

    # Bottom-left corner of of top/left overlap triangle at y0
    #y0 = int(Hi[1, 2] / Hi[2, 2])

    return p0, right, p1, (ymin, ymax)
 */

    std::tuple<GpuImage<3, Npp8u>, GpuImage<3, Npp8u>, GpuImage<3, Npp8u>> partition(
            const GpuImage<3, Npp8u> &input, const Homography &homography){
        auto partitions = homography.partitions(input.size());

        //auto left = input.slice(nppiCopy_8u_C3R, NppiRect{0, 0, static_cast<int>(p.left.x0), static_cast<int>(input.height())});
        //auto overlap = input.slice(nppiCopy_8u_C3R, NppiRect{static_cast<int>(p.left.x0), 0, static_cast<int>(p.left.x1) - static_cast<int>(p.left.x0), static_cast<int>(input.height())});
        //auto right = input.slice(nppiCopy_8u_C3R, NppiRect{static_cast<int>(p.left.x0), 0, input.swidth() - static_cast<int>(p.left.x0), static_cast<int>(input.height())});

        return std::make_tuple<GpuImage<3, Npp8u>, GpuImage<3, Npp8u>, GpuImage<3, Npp8u>>(
                // Left
                input.slice(nppiCopy_8u_C3R,
                        NppiRect{0,
                                 0,
                                 static_cast<int>(partitions.left.x0),
                                 static_cast<int>(input.height())}),
                // Overlap
                input.slice(nppiCopy_8u_C3R,
                        NppiRect{static_cast<int>(partitions.left.x0),
                                 0,
                                 static_cast<int>(partitions.left.x1) - static_cast<int>(partitions.left.x0),
                                 static_cast<int>(input.height())}),
                // Right
                input.slice(nppiCopy_8u_C3R,
                        NppiRect{static_cast<int>(partitions.left.x0),
                                 0,
                                 input.swidth() - static_cast<int>(partitions.left.x0),
                                 static_cast<int>(input.height())}));
                //GpuImage<3, Npp8u>(nppiMalloc_8u_C3, 100, 100),
                //GpuImage<3, Npp8u>(nppiMalloc_8u_C3, 100, 100),
                //GpuImage<3, Npp8u>(nppiMalloc_8u_C3, 100, 100));
    }

    void temp(const GpuImage<3, Npp8u> &input, GpuImage<3, Npp8u> &output, const Homography& homography) {
        if(nppiWarpPerspective_8u_C3R(input.device(), input.size(), input.step(), input.extent(),
                                      output.device(), output.step(), output.extent(),
                                      homography.matrix().data(),
                                      NPPI_INTER_NN) != NPP_SUCCESS)
            throw std::runtime_error("Projection failed");
    }

    void project(const GpuImage<3, Npp8u> &input, GpuImage<3, Npp8u> &output, const Homography& homography) {
        GpuImage<3, Npp8u> o1{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o2{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o3{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o4{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o5{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o6{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o7{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o8{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o9{nppiMalloc_8u_C3, output.height(), output.width()};
        GpuImage<3, Npp8u> o10{nppiMalloc_8u_C3, output.height(), output.width()};

        auto start = std::chrono::high_resolution_clock::now();
        /*temp(input, o1, homography);
        temp(input, o2, homography);
        temp(input, o3, homography);
        temp(input, o4, homography);
        temp(input, o5, homography);
        temp(input, o6, homography);
        temp(input, o7, homography);
        temp(input, o8, homography);
        temp(input, o9, homography);
        temp(input, o10, homography);*/
        for(auto i = 0u; i < 100; i++)
            partition(input, homography);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        printf("ms %lu\n", duration.count());

        /*if(nppiWarpPerspective_8u_C3R(input.device(), input.size(), input.step(), input.extent(),
                                  output.device(), output.step(), output.extent(),
                                  homography.matrix().data(),
                                  NPPI_INTER_NN) != NPP_SUCCESS)
        throw std::runtime_error("Projection failed");*/
    }
}