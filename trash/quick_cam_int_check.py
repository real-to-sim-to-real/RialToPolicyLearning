import pyrealsense2 as rs
from typing import List
import time
import numpy as np


def enable_devices(serials: List[str], ctx: rs.context, resolution_width: int=640, resolution_height: int=480, frame_rate: int=30):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, frame_rate)
        pipe.start(cfg)
        time.sleep(1.0)
        pipelines.append([serial,pipe])

    return pipelines


def pipeline_stop(pipelines: List[rs.pipeline]):
    for (device, pipe) in pipelines:
        # Stop streaming
        pipe.stop()

serial_no = '843112073228'

serial_list = [serial_no]

ctx = rs.context() # Create librealsense context for managing devices

# Define some constants
resolution_width = 640 # pixels
resolution_height = 480 # pixels
frame_rate = 30  # fps


pipelines = enable_devices(serial_list, ctx, resolution_width, resolution_height, frame_rate)

device, pipe = pipelines[0]

frames = pipe.wait_for_frames(100)

aligned_frames = align.process(frames)

# Get aligned frames
aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
color_frame = aligned_frames.get_color_frame()

depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

intrinsics_matrix = np.array(
    [[depth_intrin.fx, 0., depth_intrin.ppx],
    [0., depth_intrin.fy, depth_intrin.ppy],
    [0., 0., 1.]]
)

print(f'Intrinsics: {intrinsics_matrix}')
from IPython import embed; embed()
