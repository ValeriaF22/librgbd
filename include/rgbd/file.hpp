#pragma once

#include "camera_calibration.hpp"
#include "plane.hpp"

#pragma warning(push)
#pragma warning(disable : 4245 4267 4828 6387 26495 26812)
#include <ebml/EbmlContexts.h>
#include <ebml/EbmlCrc32.h>
#include <ebml/EbmlHead.h>
#include <ebml/EbmlStream.h>
#include <ebml/EbmlSubHead.h>
#include <ebml/EbmlVoid.h>
#include <ebml/MemReadIOCallback.h>
#include <ebml/StdIOCallback.h>
#include <matroska/FileKax.h>
#include <matroska/KaxBlockData.h>
#include <matroska/KaxCluster.h>
#include <matroska/KaxContexts.h>
#include <matroska/KaxCuesData.h>
#include <matroska/KaxInfoData.h>
#include <matroska/KaxSeekHead.h>
#include <matroska/KaxSegment.h>
#include <matroska/KaxTracks.h>
#pragma warning(pop)

namespace rgbd
{
enum class FileFrameType
{
    RGBD = 0,
    Audio = 1
};

class FileFrame
{
public:
    virtual ~FileFrame() {}
    virtual FileFrameType getType() = 0;
};

class FileVideoFrame : public FileFrame
{
public:
    FileVideoFrame(int64_t global_timecode,
                   const Bytes& color_bytes,
                   const Bytes& depth_bytes,
                   const optional<Bytes>& depth_confidence_bytes,
                   const Plane& floor)
        : global_timecode_{global_timecode}
        , color_bytes_{color_bytes}
        , depth_bytes_{depth_bytes}
        , depth_confidence_bytes_{depth_confidence_bytes}
        , floor_{floor}
    {
    }
    FileFrameType getType()
    {
        return FileFrameType::RGBD;
    }
    int64_t global_timecode() const noexcept
    {
        return global_timecode_;
    }
    const Bytes& color_bytes() const noexcept
    {
        return color_bytes_;
    }
    const Bytes& depth_bytes() const noexcept
    {
        return depth_bytes_;
    }
    const optional<Bytes>& depth_confidence_bytes() const noexcept
    {
        return depth_confidence_bytes_;
    }
    const Plane& floor() const noexcept
    {
        return floor_;
    }

private:
    int64_t global_timecode_;
    Bytes color_bytes_;
    Bytes depth_bytes_;
    optional<Bytes> depth_confidence_bytes_;
    Plane floor_;
};

class FileAudioFrame : public FileFrame
{
public:
    FileAudioFrame(int64_t global_timecode,
                     const Bytes& bytes)
        : global_timecode_{global_timecode}
        , bytes_{bytes}
    {
    }
    FileFrameType getType()
    {
        return FileFrameType::Audio;
    }
    int64_t global_timecode() const noexcept
    {
        return global_timecode_;
    }
    const Bytes& bytes() const noexcept
    {
        return bytes_;
    }

private:
    int64_t global_timecode_;
    Bytes bytes_;
};

class File
{
public:
    File(const shared_ptr<CameraCalibration>& camera_calibration,
         vector<unique_ptr<FileVideoFrame>>&& video_frames,
         vector<unique_ptr<FileAudioFrame>>&& audio_frames);
    CameraCalibration* camera_calibration() noexcept
    {
        return camera_calibration_.get();
    }
    const CameraCalibration* camera_calibration() const noexcept
    {
        return camera_calibration_.get();
    }
    const vector<unique_ptr<FileVideoFrame>>& video_frames() const noexcept
    {
        return video_frames_;
    }
    const vector<unique_ptr<FileAudioFrame>>& audio_frames() const noexcept
    {
        return audio_frames_;
    }

private:
    shared_ptr<CameraCalibration> camera_calibration_;
    vector<unique_ptr<FileVideoFrame>> video_frames_;
    vector<unique_ptr<FileAudioFrame>> audio_frames_;
};

} // namespace tg
