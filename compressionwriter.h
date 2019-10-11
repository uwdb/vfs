#ifndef VFS_COMPRESSIONWRITER_H
#define VFS_COMPRESSIONWRITER_H

#include "jointcompression.h"
#include "VideoEncoderSession.h"
#include "homography.h"
//#include "dynlink_nvcuvid.h"

namespace vfs {

    class CompressionWriter { //: public VideoWriter {
    public:
        explicit CompressionWriter(std::filesystem::path path, const size_t gop_size)
            : path_{std::move(path)},
              context_{0},
              lock_{context_},
              configuration_{Configuration{960, 540, 0, 0, 0, {30, 1}, {0, 0}}, EncodeCodec::NV_ENC_HEVC, gop_size},
              encoder_{context_, configuration_, lock_, NV_ENC_BUFFER_FORMAT_YV12},
              gpu_frame_{configuration_.height, configuration_.width, NV_ENC_PIC_STRUCT_FRAME, NV_ENC_BUFFER_FORMAT_YV12},
              plane_offsets_{/* y */ reinterpret_cast<Npp8u*>(gpu_frame_.handle()),
                             /* v */ reinterpret_cast<Npp8u*>(gpu_frame_.handle() + (5 * configuration_.height / 4) * gpu_frame_.pitch()),
                             /* u */ reinterpret_cast<Npp8u*>(gpu_frame_.handle() + configuration_.height * gpu_frame_.pitch())},
              plane_pitches_{/* y */ static_cast<int>(gpu_frame_.pitch()),
                             /* v */ static_cast<int>(gpu_frame_.pitch() / 2),
                             /* u */ static_cast<int>(gpu_frame_.pitch() / 2)},
              gop_{0}
        { }

        void write(const graphics::GpuImage<3> &frame) {
            //CudaFrame cuda{frame.height(), frame.width(), NV_ENC_PIC_STRUCT_FRAME, reinterpret_cast<CUdeviceptr>(frame.device()), frame.step()};
            //CudaFrame cuda{frame.height(), frame.width(), NV_ENC_PIC_STRUCT_FRAME, NV_ENC_BUFFER_FORMAT_YV12};
            //CudaFrame temp{frame.height() * 4, frame.width(), NV_ENC_PIC_STRUCT_FRAME};//TODO

            //CUdeviceptr u, v;
            //size_t up, vp;
            //cuMemAllocPitch(&u, &up, frame.width(), frame.height() / 2, 8);
            //cuMemAllocPitch(&v, &vp, frame.width(), frame.height() / 2, 8);

            //cudaMemset2D((void*)cuda.handle(), frame.step(), 64, frame.width(), frame.height());
            //cudaError r1 = cudaMemset2D((void*)cuda.handle(), cuda.pitch(), 0, frame.width() , frame.height());
            //auto ustart = (void*)cuda.handle() + (frame.height()-0) * cuda.pitch();
            //cudaError r2 = cudaMemset2D(ustart, cuda.pitch(), 200, cuda.pitch(), frame.height() / 4);
            //auto vstart = ustart + (frame.height() / 4) * cuda.pitch();
            //cudaError r3 = cudaMemset2D(vstart, cuda.pitch(), 255, cuda.pitch(), frame.height() / 4);
            //cudaError r2 = cudaMemset2D((void*)cuda.handle() + cuda.height() * cuda.pitch(), cuda.pitch(), 127, cuda.width() / 2, cuda.height());
            //assert(r1 == CUDA_SUCCESS);
            //assert(r2 == CUDA_SUCCESS);
            //assert(r3 == CUDA_SUCCESS);

            /*CUdeviceptr d;
            cuMemAlloc(&d, 3/2 * frame.height() * frame.width());
            cudaMemset2D((void*)d, frame.width(), 0, frame.height() * 3/2, frame.width());
            CudaFrame cuda{frame.height(), frame.width(), NV_ENC_PIC_STRUCT_FRAME, d, frame.width()};
*/
            if(session_ == nullptr || session_->frameCount() == encoder_.configuration().gopLength) {
                gop_++;
                session_ = nullptr;
                writer_ = std::make_unique<FileEncodeWriter>(encoder_, path_ / (std::to_string(gop_ - 1) + ".h264"));
                session_ = std::make_unique<VideoEncoderSession>(encoder_, *writer_);

                //if(gop_ == 2)
                //    exit(1);
            }

            assert(frame.width() == gpu_frame_.width() &&
                   frame.height() == gpu_frame_.height());
            //printf("fc %d %d\n", session_->frameCount(), encoder_.configuration().gopLength);

            //auto ustart = (void*)gpu_frame_.handle() + (frame.height()-0) * gpu_frame_.pitch();
            //auto vstart = ustart + (frame.height() / 4) * gpu_frame_.pitch();

            //NppiSize size{frame.swidth(), frame.sheight()};
            //auto uoffset = (frame.height() - 0) * cuda.pitch();
            //auto voffset = uoffset + (frame.height()/2) * cuda.pitch();
            //auto voffset = uoffset + ((frame.height()-1)/2) * cuda.pitch();
            //auto voffset = uoffset + cuda.pitch() / 2;
                    //frame.height()/2 * cuda.pitch() + frame.width() / 2;
            //Npp8u *out[3]{
            //    reinterpret_cast<Npp8u*>(gpu_frame_.handle()),
            //    reinterpret_cast<Npp8u*>(vstart),
            //    reinterpret_cast<Npp8u*>(ustart)
            //    //reinterpret_cast<Npp8u*>(temp.handle() + uoffset),
            //    //reinterpret_cast<Npp8u*>(temp.handle() + voffset)
            //};
//            int steps[3]{
  //                  gpu_frame_.pitch(), gpu_frame_.pitch()/2, gpu_frame_.pitch()/2};
                    //cuda.pitch(), cuda.pitch()/2, cuda.pitch()/2};
            //auto result =
            //auto result =
            //nppiRGBToYUV_8u_C3P3R(
            assert(nppiRGBToYUV420_8u_C3P3R(
                    reinterpret_cast<const Npp8u*>(frame.device()),
                    frame.step(),
                    plane_offsets_,
                    plane_pitches_, //cuda.pitch(), //steps, //cuda.pitch(), //steps,
                    {frame.swidth(), frame.sheight()})
            //auto c1 = cudaMemcpy(ustart, (void*)u, (cuda.pitch() * cuda.height()) / 4, cudaMemcpyDeviceToDevice);
            //auto c1 = cudaMemcpy2D((void*)cuda.handle() + uoffset, cuda.pitch(), (void*)u, up, frame.width(), frame.height() / 2, cudaMemcpyDeviceToDevice);
            //assert(c1 == CUDA_SUCCESS);
            /*nppiRGBToYUV_8u_C3R(
                    reinterpret_cast<const Npp8u*>(frame.device()),
                    frame.step(),
                    reinterpret_cast<Npp8u*>(cuda.handle()),
                    cuda.pitch(),
                    size);*/
            //assert(result
            == NPP_SUCCESS);

            session_->Encode(gpu_frame_);
            //session_->Flush(); //TODO
        }

        void flush() /* override*/ {
            if(session_ != nullptr)
                session_->Flush();
        }

        //void write(const std::vector<unsigned char>::iterator &left,
        //           const std::vector<unsigned char>::iterator &right) override {
        //}

    private:
        const std::filesystem::path path_;
        GPUContext context_;
        VideoLock lock_;
        EncodeConfiguration configuration_;
        VideoEncoder encoder_;
        CudaFrame gpu_frame_;
        Npp8u *plane_offsets_[3];
        int plane_pitches_[3];
        size_t gop_;
        std::unique_ptr<FileEncodeWriter> writer_;
        std::unique_ptr<VideoEncoderSession> session_;
    };

} // namespace vfs
#endif //VFS_COMPRESSIONWRITER_H
