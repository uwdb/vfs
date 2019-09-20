#ifndef VFS_JOINTCOMPRESSION_H
#define VFS_JOINTCOMPRESSION_H

#include "virtualvideo.h"
#include "homography.h"
#include "projection.h"

namespace vfs::graphics {
template<size_t channels>
class HomographyUpdateStrategy;

class VideoWriter {
public:
    virtual void write(const std::vector<unsigned char>::iterator&,
                       const std::vector<unsigned char>::iterator&) = 0;
};

template<size_t channels>
class JointWriter: public VideoWriter {
public:
    template<typename HomographyUpdateStrategy>
    JointWriter(size_t height, size_t width, const HomographyUpdateStrategy &homography_update)
        : left_frame_(nppiMalloc_8u_C3, height, width),
          right_frame_(nppiMalloc_8u_C3, height, width),
          configuration_{height, width},
          partitions_{},
          homography_update_{std::make_unique<HomographyUpdateStrategy>(homography_update)}
    { }

    void write(const std::vector<unsigned char>::iterator &left,
               const std::vector<unsigned char>::iterator &right) override {
        homography_update_->update(*this);

        assert(partitions_);

        left_frame_.upload(left);
        right_frame_.upload(right);

        graphics::partition(left_frame_, right_frame_, *partitions_);
        graphics::project(right_frame_, partitions_->overlap(), partitions_->homography());
    }

    const GpuImage<channels, Npp8u>& left_frame() const { return left_frame_; }
    const GpuImage<channels, Npp8u>& right_frame() const { return right_frame_; }

    bool has_homography() const { return partitions_ != nullptr; }
    void homography(const Homography & homography) {
        if(partitions_ == nullptr || homography != partitions_->homography())
            partitions_ = std::make_unique<PartitionBuffer>(left_frame_, homography);
    }
    SiftConfiguration& configuration() { return configuration_; }

private:
    GpuImage<channels, Npp8u> left_frame_;
    GpuImage<channels, Npp8u> right_frame_;

    SiftConfiguration configuration_;
    std::unique_ptr<PartitionBuffer> partitions_;
    std::unique_ptr<HomographyUpdateStrategy<channels>> homography_update_;
};

template<size_t channels>
class HomographyUpdateStrategy {
public:
    virtual void update(JointWriter<channels>&) = 0;
};

class OneTimeHomographyUpdate : public HomographyUpdateStrategy<3> {
public:
    void update(JointWriter<3> &writer) override {
        if(!writer.has_homography())
            writer.homography(
                    find_homography(writer.left_frame(), writer.right_frame(),
                                    writer.configuration()));
    }
};

} // namespace vfs

#endif //VFS_JOINTCOMPRESSION_H
