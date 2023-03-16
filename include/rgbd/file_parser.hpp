#pragma once

#include "file.hpp"
#include "ios_camera_calibration.hpp"
#include "kinect_camera_calibration.hpp"

namespace rgbd
{

class FileParser
{
public:
    FileParser(const void* ptr, size_t size);
    FileParser(const string& file_path);

private:
    void parseExceptClusters();
    optional<const FileInfo> parseInfo(unique_ptr<libmatroska::KaxInfo>& kax_info);
    optional<const FileOffsets> parseOffsets(unique_ptr<libmatroska::KaxSegment>& segment);
    optional<const FileTracks> parseTracks(unique_ptr<libmatroska::KaxTracks>& tracks);
    optional<const FileAttachments>
    parseAttachments(unique_ptr<libmatroska::KaxAttachments>& attachments);
    FileFrame* parseCluster(unique_ptr<libmatroska::KaxCluster>& cluster);
    void parseAllClusters(vector<FileVideoFrame>& video_frames,
                          vector<FileAudioFrame>& audio_frames,
                          vector<FileIMUFrame>& imu_frames,
                          vector<FileTRSFrame>& trs_frames);

public:
    unique_ptr<File> parse(bool with_frames, bool with_directions);

private:
    unique_ptr<libebml::IOCallback> input_;
    EbmlStream stream_;
    unique_ptr<libmatroska::KaxSegment> kax_segment_;
    optional<FileOffsets> file_offsets_;
    optional<FileInfo> file_info_;
    optional<FileTracks> file_tracks_;
    optional<FileAttachments> file_attachments_;
};
} // namespace rgbd
