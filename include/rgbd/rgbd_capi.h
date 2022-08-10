#pragma once

#ifdef __cplusplus
extern "C"
{
#endif
#include <stdint.h>
#include <stdlib.h>

    //////// START ENUMS ////////
    typedef enum
    {
        RGBD_CAMERA_DEVICE_TYPE_AZURE_KINECT = 0,
        RGBD_CAMERA_DEVICE_TYPE_IOS = 1
    } rgbdCameraDeviceType;

    typedef enum
    {
        RGBD_COLOR_CODEC_TYPE_VP8 = 0
    } rgbdColorCodecType;

    typedef enum
    {
        RGBD_FILE_FRAME_TYPE_VIDEO = 0,
        RGBD_FILE_FRAME_TYPE_AUDIO = 1
    } rgbdFileFrameType;
    //////// END ENUMS ////////

    //////// START CONSTANTS ////////
    int RGBD_MAJOR_VERSION();
    int RGBD_MINOR_VERSION();
    int RGBD_PATCH_VERSION();
    int RGBD_AUDIO_SAMPLE_RATE();
    int RGBD_AUDIO_INPUT_CHANNEL_COUNT();
    int RGBD_AUDIO_INPUT_SAMPLES_PER_FRAME();
    int RGBD_AUDIO_OUTPUT_CHANNEL_COUNT();
    int RGBD_AUDIO_OUTPUT_INTERVAL_SECONDS_RECIPROCAL();
    int RGBD_AUDIO_OUTPUT_SAMPLES_PER_FRAME();
    //////// END CONSTANTS ////////

    //////// START CAPI UTILITY CLASSES ////////
    void rgbd_cbyte_array_dtor(void* ptr);
    uint8_t* rgbd_cbyte_array_data(void* ptr);
    size_t rgbd_cbyte_array_size(void* ptr);

    void rgbd_cfloat_array_dtor(void* ptr);
    float* rgbd_cfloat_array_data(void* ptr);
    size_t rgbd_cfloat_array_size(void* ptr);

    void rgbd_cint16_array_dtor(void* ptr);
    int16_t* rgbd_cint16_array_data(void* ptr);
    size_t rgbd_cint16_array_size(void* ptr);

    void rgbd_cuint8_array_dtor(void* ptr);
    uint8_t* rgbd_cuint8_array_data(void* ptr);
    size_t rgbd_cuint8_array_size(void* ptr);

    void rgbd_cstring_dtor(void* ptr);
    const char* rgbd_cstring_c_str(void* ptr);
    //////// END CAPI UTILITY CLASSES ////////

    //////// START CAMERA CALIBRATION ////////
    void rgbd_camera_calibration_dtor(void* ptr);
    rgbdCameraDeviceType rgbd_camera_calibration_get_camera_device_type(void* ptr);
    //////// START CAMERA CALIBRATION ////////

    //////// START FFMPEG AUDIO DECODER ////////
    void* rgbd_ffmpeg_audio_decoder_ctor();
    void rgbd_ffmpeg_audio_decoder_dtor(void* ptr);
    void* rgbd_ffmpeg_audio_decoder_decode(void* ptr,
                                         const uint8_t* opus_frame_data,
                                         size_t opus_frame_size);
    //////// END FFMPEG AUDIO DECODER ////////

    //////// START FFMPEG VIDEO DECODER ////////
    void* rgbd_ffmpeg_video_decoder_ctor(rgbdColorCodecType type);
    void rgbd_ffmpeg_video_decoder_dtor(void* ptr);
    void* rgbd_ffmpeg_video_decoder_decode(void* ptr,
                                           const uint8_t* vp8_frame_data,
                                           size_t vp8_frame_size);
    //////// END FFMPEG VIDEO DECODER ////////

    //////// START FILE ////////
    void rgbd_file_dtor(void* ptr);
    void* rgbd_file_get_attachments(void* ptr);
    size_t rgbd_file_get_video_frame_count(void* ptr);
    void* rgbd_file_get_video_frame(void* ptr, size_t index);
    size_t rgbd_file_get_audio_frame_count(void* ptr);
    void* rgbd_file_get_audio_frame(void* ptr, size_t index);
    //////// END FILE ////////

    //////// START FILE ATTACHMENTS ////////
    void rgbd_file_attachments_dtor(void* ptr);
    void* rgbd_file_attachments_get_camera_calibration(void* ptr);
    void* rgbd_file_attachments_get_cover_png_bytes(void* ptr);
    //////// END FILE ATTACHMENTS ////////

    //////// START FILE AUDIO FRAME ////////
    void rgbd_file_audio_frame_dtor(void* ptr);
    int64_t rgbd_file_audio_frame_get_global_timecode(void* ptr);
    void* rgbd_file_audio_frame_get_bytes(void* ptr);
    //////// END FILE AUDIO FRAME ////////

    //////// START FILE FRAME ////////
    void rgbd_file_frame_dtor(void* ptr);
    rgbdFileFrameType rgbd_file_frame_get_type(void* ptr);
    //////// END FILE FRAME ////////

    //////// START FILE PARSER ////////
    void* rgbd_file_parser_ctor_from_data(void* ptr, size_t size);
    void* rgbd_file_parser_ctor_from_path(const char* file_path);
    void rgbd_file_parser_dtor(void* ptr);
    double rgbd_file_parser_get_duration_us(void* ptr);
    void* rgbd_file_parser_get_writing_app(void* ptr);
    rgbdCameraDeviceType rgbd_file_parser_get_camera_device_type(void* ptr);
    void* rgbd_file_parser_get_camera_calibration(void* ptr);
    void* rgbd_file_parser_get_color_track_codec(void* ptr);
    int rgbd_file_parser_get_color_track_width(void* ptr);
    int rgbd_file_parser_get_color_track_height(void* ptr);
    void* rgbd_file_parser_get_depth_track_codec(void* ptr);
    int rgbd_file_parser_get_depth_track_width(void* ptr);
    int rgbd_file_parser_get_depth_track_height(void* ptr);
    void* rgbd_file_parser_get_cover_png_bytes(void* ptr);
    void* rgbd_file_parser_parse_no_frames(void* ptr);
    void* rgbd_file_parser_parse_all_frames(void* ptr);
    //////// END FILE PARSER ////////

    //////// START FILE VIDEO FRAME ////////
    void rgbd_file_video_frame_dtor(void* ptr);
    int64_t rgbd_file_video_frame_get_global_timecode(void* ptr);
    void* rgbd_file_video_frame_get_color_bytes(void* ptr);
    void* rgbd_file_video_frame_get_depth_bytes(void* ptr);
    float rgbd_file_video_frame_get_floor_normal_x(void* ptr);
    float rgbd_file_video_frame_get_floor_normal_y(void* ptr);
    float rgbd_file_video_frame_get_floor_normal_z(void* ptr);
    float rgbd_file_video_frame_get_floor_constant(void* ptr);
    //////// END FILE VIDEO FRAME ////////

    //////// START KINECT CAMERA CALIBRATION ////////
    void* rgbd_kinect_camera_calibration_ctor(int color_width,
                                              int color_height,
                                              int depth_width,
                                              int depth_height,
                                              int resolution_width,
                                              int resolution_height,
                                              float cx,
                                              float cy,
                                              float fx,
                                              float fy,
                                              float k1,
                                              float k2,
                                              float k3,
                                              float k4,
                                              float k5,
                                              float k6,
                                              float codx,
                                              float cody,
                                              float p1,
                                              float p2,
                                              float max_radius_for_projection);
    int rgbd_kinect_camera_calibration_get_color_width(void* ptr);
    int rgbd_kinect_camera_calibration_get_color_height(void* ptr);
    int rgbd_kinect_camera_calibration_get_depth_width(void* ptr);
    int rgbd_kinect_camera_calibration_get_depth_height(void* ptr);
    int rgbd_kinect_camera_calibration_get_resolution_width(void* ptr);
    int rgbd_kinect_camera_calibration_get_resolution_height(void* ptr);
    float rgbd_kinect_camera_calibration_get_cx(void* ptr);
    float rgbd_kinect_camera_calibration_get_cy(void* ptr);
    float rgbd_kinect_camera_calibration_get_fx(void* ptr);
    float rgbd_kinect_camera_calibration_get_fy(void* ptr);
    float rgbd_kinect_camera_calibration_get_k1(void* ptr);
    float rgbd_kinect_camera_calibration_get_k2(void* ptr);
    float rgbd_kinect_camera_calibration_get_k3(void* ptr);
    float rgbd_kinect_camera_calibration_get_k4(void* ptr);
    float rgbd_kinect_camera_calibration_get_k5(void* ptr);
    float rgbd_kinect_camera_calibration_get_k6(void* ptr);
    float rgbd_kinect_camera_calibration_get_codx(void* ptr);
    float rgbd_kinect_camera_calibration_get_cody(void* ptr);
    float rgbd_kinect_camera_calibration_get_p1(void* ptr);
    float rgbd_kinect_camera_calibration_get_p2(void* ptr);
    float rgbd_kinect_camera_calibration_get_max_radius_for_projection(void* ptr);
    //////// END KINECT CAMERA CALIBRATION ////////

    //////// START INT16 FRAME ////////
    void rgbd_int16_frame_dtor(void* ptr);
    int rgbd_int16_frame_get_width(void* ptr);
    int rgbd_int16_frame_get_height(void* ptr);
    void* rgbd_int16_frame_get_values(void* ptr);
    //////// END INT16 FRAME ////////

    //////// START IOS CAMERA CALIBRATION ////////
    void* rgbd_ios_camera_calibration_ctor(int color_width,
                                           int color_height,
                                           int depth_width,
                                           int depth_height,
                                           float fx,
                                           float fy,
                                           float ox,
                                           float oy,
                                           float reference_dimension_width,
                                           float reference_dimension_height,
                                           float lens_distortion_center_x,
                                           float lens_distortion_center_y,
                                           const float* lens_distortion_lookup_table,
                                           size_t lens_distortion_lookup_table_size);
    int rgbd_ios_camera_calibration_get_color_width(void* ptr);
    int rgbd_ios_camera_calibration_get_color_height(void* ptr);
    int rgbd_ios_camera_calibration_get_depth_width(void* ptr);
    int rgbd_ios_camera_calibration_get_depth_height(void* ptr);
    float rgbd_ios_camera_calibration_get_fx(void* ptr);
    float rgbd_ios_camera_calibration_get_fy(void* ptr);
    float rgbd_ios_camera_calibration_get_ox(void* ptr);
    float rgbd_ios_camera_calibration_get_oy(void* ptr);
    float rgbd_ios_camera_calibration_get_reference_dimension_width(void* ptr);
    float rgbd_ios_camera_calibration_get_reference_dimension_height(void* ptr);
    float rgbd_ios_camera_calibration_get_lens_distortion_center_x(void* ptr);
    float rgbd_ios_camera_calibration_get_lens_distortion_center_y(void* ptr);
    void* rgbd_ios_camera_calibration_get_lens_distortion_lookup_table(void* ptr);
    //////// END IOS CAMERA CALIBRATION ////////

    //////// START RECORDER ////////
    void* rgbd_recorder_ctor(const char* file_path,
                             bool has_depth_confidence,
                             void* calibration,
                             int color_bitrate,
                             int framerate,
                             int depth_diff_multiplier,
                             int samplerate);
    void rgbd_recorder_dtor(void* ptr);
    void rgbd_recorder_record_rgbd_frame(void* ptr,
                                       int64_t time_point_us,
                                       int width,
                                       int height,
                                       const uint8_t* y_channel,
                                       size_t y_channel_size,
                                       const uint8_t* u_channel,
                                       size_t u_channel_size,
                                       const uint8_t* v_channel,
                                       size_t v_channel_size,
                                       const int16_t* depth_values,
                                       size_t depth_values_size,
                                       const uint8_t* depth_confidence_values,
                                       size_t depth_confidence_values_size,
                                       float floor_normal_x,
                                       float floor_normal_y,
                                       float floor_normal_z,
                                       float floor_distance);
    void rgbd_recorder_record_audio_frame(void* ptr,
                                        int64_t time_point_us,
                                        const float* pcm_samples,
                                        size_t pcm_samples_size);
    void rgbd_recorder_record_flush(void* ptr);
    //////// END RECORDER ////////

    //////// START TDC1 DECODER ////////
    void* rgbd_tdc1_decoder_ctor();
    void rgbd_tdc1_decoder_dtor(void* ptr);
    void* rgbd_tdc1_decoder_decode(void* ptr,
                                 const uint8_t* encoded_depth_frame_data,
                                 size_t encoded_depth_frame_size);
    //////// END TDC1 DECODER ////////

    //////// START YUV FRAME ////////
    void rgbd_yuv_frame_dtor(void* ptr);
    void* rgbd_yuv_frame_get_y_channel(void* ptr);
    void* rgbd_yuv_frame_get_u_channel(void* ptr);
    void* rgbd_yuv_frame_get_v_channel(void* ptr);
    int rgbd_yuv_frame_get_width(void* ptr);
    int rgbd_yuv_frame_get_height(void* ptr);
    //////// END YUV FRAME ////////
#ifdef __cplusplus
}
#endif
