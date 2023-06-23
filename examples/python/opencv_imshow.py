import pyrgbd as rgbd
import argparse
import examples_utils
import numpy as np
import cv2
import os

def main():
    parser = argparse.ArgumentParser(
        prog="pyrgbd example imshow",
        description="pyrgbd example of showing video frames using OpenCV.",
    )
    parser.add_argument("-i", "--input", help="path to input 3D video file")

    args = parser.parse_args()
    # if args.input is None:
    #     print(
    #         "No input file path specified. You can download a sample video from https://telegie.com/posts/A9oofdweNJ4"
    #     )
    #     return
    
    # path of the mkv file
    args.input = "cat.mkv"

    # Specify the output directory path
    output_dir = "/Users/user/Desktop/mkv_results/librgbd_results/0010_K_R"
    
    # Parse the video file (record).
    record_parser = rgbd.RecordParser(args.input)
    record = record_parser.parse(True, True)

    # Print basic codec info.
    record_tracks = record.get_tracks()
    print(f"color width: {record_tracks.color_track.width}")
    print(f"color height: {record_tracks.color_track.height}")
    print(f"color codec: {record_tracks.color_track.codec}")
    print(f"depth width: {record_tracks.depth_track.width}")
    print(f"depth height: {record_tracks.depth_track.height}")
    print(f"depth codec: {record_tracks.depth_track.codec}")

    # Obtain encoded video frames.
    record_video_frames = record.get_video_frames()

    # Decode color video frames.
    yuv_frames = []
    color_decoder = rgbd.ColorDecoder(record_tracks.color_track.codec)
    for video_frame in record_video_frames:
        yuv_frames.append(color_decoder.decode(video_frame.get_color_bytes()))

    # Decode depth video frames.
    depth_frames = []
    depth_decoder = rgbd.DepthDecoder(record_tracks.depth_track.codec)
    for video_frame in record_video_frames:
        depth_frames.append(depth_decoder.decode(video_frame.get_depth_bytes()))

    # Show frames using OpenCV.
    for index in range(len(record_video_frames)):
        yuv_frame = yuv_frames[index]
        rgb_array = examples_utils.convert_yuv_frame_to_rgb_array(yuv_frame)

        depth_frame = depth_frames[index]
        depth_array = depth_frame.get_values().astype(np.uint16)

        # Improve visibility when shown via cv2.imshow.
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)
        depth_array = depth_array * 20
        depth_map_path = os.path.join(output_dir, f"depth_map_{index}.png")

        # Save depth map as an image using OpenCV.
        cv2.imwrite(depth_map_path, depth_array)

        print(f"Depth map {index} saved at {depth_map_path}")    

        cv2.imshow("color", rgb_array)
        cv2.imshow("depth", depth_array)
        cv2.waitKey(1)

    # Define output video path and parameters
    output_path = output_dir + "/rgb.mp4"
    fps = 60.0  # Frames per second
    width = record_tracks.color_track.width
    height = record_tracks.color_track.height

    # Define the codec for the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Create VideoWriter object
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames to the output video
    for index in range(len(record_video_frames)):
        yuv_frame = yuv_frames[index]
        rgb_array = examples_utils.convert_yuv_frame_to_rgb_array(yuv_frame)

        # Improve visibility when shown via cv2.imshow.
        rgb_array = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2RGB)

        # Write the frame to the output video
        video_writer.write(rgb_array)

    # Release the VideoWriter object
    video_writer.release()

    print(f"Video saved at {output_path}")

if __name__ == "__main__":
    main()


