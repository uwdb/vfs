#ifndef VFS_HOMOGRAPHY_H
#define VFS_HOMOGRAPHY_H

#include <memory>
#include <array>
#include <functional>
#include "cuda_runtime.h"
#include "nppi.h"

#include "/home/bhaynes/projects/eigen/Eigen/Dense"

class CudaImage;

namespace vfs::graphics {
    namespace internal {
        struct SiftData;
    }

    template<const size_t channels, typename... T>
    class GpuImage;

    template<const size_t channels>
    class GpuImage<channels> {
    public:
        GpuImage(void* device, const int step, const size_t height, const size_t width)
            : device_(device), step_(step), height_(height), width_(width)
        { }

        size_t height() const { return height_; }
        size_t width() const { return width_; }
        int sheight() const { return static_cast<int>(height_); }
        int swidth() const { return static_cast<int>(width_); }

        void* device() const { return device_; }
        int step() const { return step_; }

        NppiSize size() const { return {static_cast<int>(width()), static_cast<int>(height())}; }
        NppiRect extent() const { return {0, 0, static_cast<int>(width()), static_cast<int>(height())}; }

    protected:
        void device(void* value) { device_ = value; }

    private:
        void* device_;
        const int step_;
        const size_t height_, width_;
    };

    template<const size_t channels, typename T>
    class GpuImage<channels, T>: public GpuImage<channels> {
    public:
        //GpuImage(const T* device, const int step, const size_t height, const size_t width)
        //        : GpuImage<channels>(static_cast<void*>(device), step, height, width), owned_(false)
        //{ }

        GpuImage(const std::function<T*(int, int, int*)> &allocator, const size_t height, const size_t width)
                : GpuImage<channels, T>(allocate(allocator, height, width), allocator, height, width)
        { }

        GpuImage(const std::vector<T> &data, const std::function<T*(int, int, int*)> &allocator, const size_t height, const size_t width)
                : GpuImage<channels, T>(allocate(allocator, height, width), allocator, height, width) {
            if(cudaMemcpy2D(GpuImage<channels>::device(), static_cast<size_t>(GpuImage<channels>::step()),
                            data.data(), sizeof(T) * channels * width,
                            sizeof(T) * channels * width, height, cudaMemcpyHostToDevice) != cudaSuccess)
                throw std::runtime_error("Error copying data to GPU");

        }

        ~GpuImage() {
            cudaFree(device());
            GpuImage<channels>::device(nullptr);
        }

        T* device() const { return static_cast<T*>(GpuImage<channels>::device()); }

        size_t byte_step() const { return sizeof(T) * channels * GpuImage<channels>::step(); }
        int sbyte_step() const { return static_cast<int>(byte_step()); }

        GpuImage<channels, T> slice(const std::function<NppStatus(const T*, int, T*, int, NppiSize)> copier, const NppiRect &region) const {
            auto slice = GpuImage<channels, T>(allocator_, GpuImage<channels>::height(), GpuImage<channels>::width());
            auto offset = region.y * GpuImage<channels>::step() + region.x + sizeof(T);
            if(copier(device() + offset, GpuImage<channels>::step(), slice.device(), slice.step(), {region.width, region.height}) != NPP_SUCCESS)
                throw std::runtime_error("Error copying during slice");
            return slice;
        }

    private:
        GpuImage(const std::pair<T*, int> &pair, const std::function<T*(int, int, int*)> allocator, const size_t height, const size_t width)
                : GpuImage<channels>(pair.first, pair.second, height, width), owned_(true), allocator_(allocator)
        { }

        std::pair<T*, int> allocate(const std::function<T*(int, int, int*)> &allocator, const size_t height, const size_t width) {
            int step;
            T* device;

            if((device = allocator(static_cast<int>(width), static_cast<int>(height), &step)) != nullptr)
                return {device, step};
            else
                throw std::runtime_error("Failed to allocate memory");
        }

        const bool owned_;
        const std::function<T*(int, int, int*)> allocator_;
    };

    struct Partitions {
        struct {
            size_t x0, x1, y0, y1;
        } left, right;
    };

    class Homography {
    public:
        explicit Homography()
                : Homography(std::array<float, 9>{})
        { }

        explicit Homography(const std::array<float, 9> &values)
                : values_(values), inverse_{}
        { }

        explicit operator float*() {
            return values_.data();
        }

        std::array<double[3], 3> matrix() const {
            std::array<double[3], 3> matrix;

            for(auto i = 0u; i < values_.size(); i++)
                matrix.at(i % 3)[i / 3] = values_.at(i);
            return matrix;
        }

        const std::array<float, 9> &inverse() const {
            if(!inverted_) {
                inverted_ = true;

                Eigen::Matrix3f M(values_.data());

                //Transpose because Eigen is column-major and we're row-major
                Eigen::Map<Eigen::Matrix3f>(inverse_.data(), M.rows(), M.cols()) = M.transpose().inverse().transpose();
            }

            return inverse_;
        }

        Partitions partitions(const NppiSize &size) const {
/*
    # Left overlap at x0
    i = np.array([0, 0, 1])
    #Hi = np.linalg.inv(H)
    H0 = Hi.dot(i)
    H0n = H0 / H0[2]
    x0 = H0n[0]
    p0 = int(x0)
 */
            const auto &Hi = inverse();
            // M[0]/[2], where M = H^{-1} \dot [0 0 1]
            auto left0 = Hi[3] / Hi[8];
            // M[0]/[2], where M = H^{-1} \dot [width 0 1]
            auto left1 = ((size.width * Hi[0]) + Hi[3]) / Hi[8];

            assert(left0 >= -0.5);
            assert(left1 >= -0.5 && left1 >= left0);

            return {{static_cast<size_t>(left0), static_cast<size_t>(left1), 0, 0},
                    {0, 0, 0, 0}};
            //std::make_pair<size_t, size_t>(static_cast<size_t>(left0), static_cast<size_t>(left1));
        }

    private:
        std::array<float, 9> values_;
        mutable std::array<float, 9> inverse_;
        mutable bool inverted_ = false;
    };

    class SiftConfiguration {
    public:
        SiftConfiguration(size_t height, size_t width,
                          float blur = 1.0f, float threshold = 3.0f, int octaves = 5, bool scale = false);
        ~SiftConfiguration();

        CudaImage& left() { return *left_; }
        CudaImage& right() { return *right_; }
        internal::SiftData& left_data() { return *left_data_; }
        internal::SiftData& right_data() { return *right_data_; }

        int octaves() const { return octaves_; }
        float blur() const { return blur_; }
        float threshold() const { return threshold_; }

        float* scratch() const { return scratch_; }

    private:
        const size_t height_, width_;
        const float blur_, threshold_; //4.5f
        const int octaves_;
        const bool scale_;
        const std::unique_ptr<CudaImage> left_, right_;
        const std::unique_ptr<internal::SiftData> left_data_, right_data_;
        float *scratch_;
    };

    Homography find_homography(const std::vector<unsigned char>&, const std::vector<unsigned char>&,
                               size_t height, size_t width);
    Homography find_homography(const std::vector<unsigned char>&, const std::vector<unsigned char>&,
                               size_t, size_t, SiftConfiguration&);
    Homography find_homography(const GpuImage<3, Npp8u>&, const GpuImage<3, Npp8u>&, SiftConfiguration&);
    Homography find_homography(const GpuImage<1, Npp8u>&, const GpuImage<1, Npp8u>&, SiftConfiguration&);
    Homography find_homography(const GpuImage<1, Npp32f>&, const GpuImage<1, Npp32f>&, SiftConfiguration&);
}

#endif //VFS_HOMOGRAPHY_H
