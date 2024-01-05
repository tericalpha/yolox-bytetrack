from moviepy.editor import VideoFileClip

def convert_to_webm(input_file, output_file, bitrate='1M'):
    clip = VideoFileClip(input_file)
    clip.write_videofile(output_file, codec='libvpx', audio_codec='libvorbis', bitrate=bitrate, preset='ultrafast')
    clip.close()

# Replace 'input.mp4' and 'output.webm' with your file paths
convert_to_webm('/home/syed/Desktop/projects/tracker/ByteTrack/pretrained/YOLOX-ByteTrack-Car-Counter/yolox-head-output_video.avi','yolox-head-output_video.webm')