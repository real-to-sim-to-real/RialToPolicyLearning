import open3d
from rdt.common import util, path_util
from rdt.perception.realsense_util import enable_devices, RealsenseInterface
from rdt.camera.simple_multicam import MultiRealsenseLocal
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
import os, os.path as osp
import sys
import argparse
import random
import signal
import threading
import numpy as np
import copy
import meshcat
import matplotlib.pyplot as plt
import trimesh
import pyrealsense2 as rs
import cv2
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

class RealSenseCamera:

    def __init__(self, cam_idx=[2]):
        rs_cfg = get_default_multi_realsense_cfg()
        serials = rs_cfg.SERIAL_NUMBERS

        rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
        depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
        info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
        pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

        prefix = rs_cfg.CAMERA_NAME_PREFIX
        camera_names = [f'{prefix}{i}' for i in range(len(serials))]
        cam_list = [camera_names[int(idx)] for idx in cam_idx]
        serial_list = [serials[int(idx)] for idx in cam_idx]

        calib_dir = osp.join(path_util.get_rdt_src(), 'robot/camera_calibration_files')
        #calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in args.cam_index]
        calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in cam_idx]
        ctx = rs.context() # Create librealsense context for managing devices

        # Define some constants 
        resolution_width = 640 # pixels
        resolution_height = 480 # pixels
        frame_rate = 30  # fps

        self.pipelines = enable_devices(serial_list, ctx, resolution_width, resolution_height, frame_rate)

        self.cams = MultiRealsenseLocal(cam_list, calib_filenames)
        self.rs_interface = RealsenseInterface()
    
    def o3d_fps(self, np_pcd, num_samples=1024):
        print(f'Running farthest point sampling with {num_samples} points')
        o3d_pcd = open3d.geometry.PointCloud()
        o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)

        o3d_pcd_ds = o3d_pcd.farthest_point_down_sample(num_samples=num_samples)
        return np.asarray(o3d_pcd_ds.points)

    def o3d_vds(self, np_pcd, voxel_size=0.0025):
        print(f'Running voxel down sampling with {voxel_size} voxel_size')
        o3d_pcd = open3d.geometry.PointCloud()
        o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)

        o3d_pcd_ds = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
        return np.asarray(o3d_pcd_ds.points)

    def get_rgb(self):
        rgbs = []
        for idx, cam in enumerate(self.cams.cams):
            rgb, depth = self.rs_interface.get_rgb_and_depth_image(self.pipelines[idx])
            rgbs.append(rgb)
        return np.concatenate(rgbs)

    def get_pcd(self, visualize=False):
        pcd_pts = []
        pcd_colors = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        proc_pcd_list = []

        for idx, cam in enumerate(self.cams.cams):
            rgb, depth = self.rs_interface.get_rgb_and_depth_image(self.pipelines[idx])

            rgb_imgs.append(rgb)

            cam_intrinsics = self.rs_interface.get_intrinsics_mat(self.pipelines[idx])
            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat
            cam_int_list.append(cam_intrinsics)
            cam_poses_list.append(cam_pose_world)

            depth = depth * 0.001
            valid = depth < 10
            valid = np.logical_and(valid, depth > 0)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth
            # depth_imgs.append(depth_valid)

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            # pcd_cam_img = pcd_cam.reshape(depth.shape[0], depth.shape[1], 3)
            valid = pcd_cam[:, 2] > 0.01
            valid = np.logical_or(valid, pcd_cam[:,2] < - 0.03)
            valid = np.logical_or(valid, pcd_cam[:,0] > 0.03)
            valid = np.logical_or(valid, pcd_cam[:,0] < - 0.03)
            valid = np.logical_or(valid, pcd_cam[:,1] > 0.03)
            valid = np.logical_or(valid, pcd_cam[:,1] < -0.03)
            pcd_cam = pcd_cam[valid]
            pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
            # import IPython
            # IPython.embed()
            # pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)
            # pcd_dict = {
            #     'world': pcd_world,
            #     'cam': pcd_cam_img,
            #     'cam_img': pcd_cam,
            #     'world_img': pcd_world_img,
            #     'cam_pose_mat': cam_pose_world
            # }

            pcd_pts.append(pcd_world)
            pcd_colors.append(rgb.reshape(-1,3)/255)
            # pcd_dict_list.append(pcd_dict)

            # kp_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
            # keypoint_list.append(kp_dict)

            # pcd_color = pcd_color_list[idx]
            # util.meshcat_pcd_show(mc_vis, pcd_world, color=pcd_color, name=f'scene/indiv_pcds/pcd_world_cam_{idx}_first')
            # util.meshcat_pcd_show(mc_vis, proc_pcd_i, color=(0, 0, 100), name=f'{mc_iter_name}/pcd_world_cam_{idx}_first')

        pcd_full = np.concatenate(pcd_pts, axis=0)
        pcd_colors_full = np.concatenate(pcd_colors, axis=0)[valid]
        # pcd_full_ds = o3d_fps(pcd_full, num_samples=min(int(pcd_full.shape[0] * 0.5), 8192))
        # pcd_full_ds = self.o3d_vds(pcd_full, voxel_size=0.0045)
        #         self.env_lower_limits = [0.1, -0.5, 0.155]
        # self.env_upper_limits = [0.9,  0.5, 1]
        # valid = pcd_full[:,2] 
        # valid = np.logical_and(valid, pcd_full[:,2] > 0.03) #0.04

        # valid = np.logical_and(valid, pcd_full[:,0] > 0)
        # valid = np.logical_and(valid, pcd_full[:,0] < 1)
        # valid = np.logical_and(valid, pcd_full[:,1] < 0.6)
        # valid = np.logical_and(valid, pcd_full[:,1] > -0.6)
        # pcd_full = pcd_full[valid]
        # pcd_colors_full = pcd_colors_full[valid]

        if visualize:
            o3d_pcd = open3d.geometry.PointCloud()
            o3d_pcd.points = open3d.utility.Vector3dVector(pcd_full)
            o3d_pcd.colors = open3d.utility.Vector3dVector(pcd_colors_full)
            open3d.visualization.draw_geometries([o3d_pcd])
        return pcd_full, pcd_colors_full
