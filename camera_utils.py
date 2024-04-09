from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, Usd, Gf, Sdf
import carb
import numpy as np
import omni
import omni.kit.app
import time
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.utils.rotations import gf_quat_to_np_array
from scipy.spatial.transform import Rotation as R
import math
from omni.isaac.orbit.utils.math import convert_quat
import scipy.spatial.transform as tf


def compute_intrinsic_matrix(prim, img):
    focal_length = prim.GetAttribute("focalLength").Get()
    horiz_aperture = prim.GetAttribute("horizontalAperture").Get()
    height, width = img.shape
    fov = 2*math.atan(horiz_aperture/(2*focal_length))
    focal_px = width*0.5 / math.tan(fov/2)
    a = focal_px
    b = width * 0.5 
    c = focal_px
    d = height * 0.5
    return np.array([[a,0,b], [0,c,d], [0,0,1]], dtype=float)


def get_intrinsics_matrix(prim_path, width, height):
    """Get intrinsic matrix for the camera attached to a specific viewport

    Args:
        viewport (Any): Handle to viewport api

    Returns:
        np.ndarray: the intrinsic matrix associated with the specified viewport
                The following image convention is assumed:
                    +x should point to the right in the image
                    +y should point down in the image
    """
    # import IPython
    # IPython.embed()
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    focal_length = prim.GetAttribute("focalLength").Get()
    horizontal_aperture = prim.GetAttribute("horizontalAperture").Get()
    vertical_aperture = prim.GetAttribute("verticalAperture").Get()
    fx = width * focal_length / horizontal_aperture
    fy = height * focal_length / vertical_aperture
    cx = width * 0.5
    cy = height * 0.5
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def compute_camera_pose(eye, target):
    eye_position = Gf.Vec3d(np.asarray(eye).tolist())
    target_position = Gf.Vec3d(np.asarray(target).tolist())
    # compute forward direction
    forward_dir = (eye_position - target_position).GetNormalized()
    # get up axis
    up_axis_token = stage_utils.get_stage_up_axis()
    if up_axis_token == UsdGeom.Tokens.y:
        # deal with degenerate case
        if forward_dir == Gf.Vec3d(0, 1, 0):
            up_axis = Gf.Vec3d(0, 0, 1)
        elif forward_dir == Gf.Vec3d(0, -1, 0):
            up_axis = Gf.Vec3d(0, 0, -1)
        else:
            up_axis = Gf.Vec3d(0, 1, 0)
    elif up_axis_token == UsdGeom.Tokens.z:
        # deal with degenerate case
        if forward_dir == Gf.Vec3d(0, 0, 1):
            up_axis = Gf.Vec3d(0, 1, 0)
        elif forward_dir == Gf.Vec3d(0, 0, -1):
            up_axis = Gf.Vec3d(0, -1, 0)
        else:
            up_axis = Gf.Vec3d(0, 0, 1)
    else:
        raise NotImplementedError(f"This method is not supported for up-axis '{up_axis_token}'.")
    # compute matrix transformation
    # view matrix: camera_T_world
    matrix_gf = Gf.Matrix4d(1).SetLookAt(eye_position, target_position, up_axis)
    # camera position and rotation in world frame
    matrix_gf = matrix_gf.GetInverse()
    position = np.asarray(matrix_gf.ExtractTranslation())
    orientation = gf_quat_to_np_array(matrix_gf.ExtractRotationQuat())

    return position, orientation
        # set camera poses using the view

def backproject_depth(prim_path, depth_image, max_clip_depth):
    """Backproject depth image to image space

    Args:
        depth_image (np.array): Depth image buffer
        viewport_api (Any): Handle to viewport api
        max_clip_depth (float): Depth values larger than this will be clipped

    Returns:
        np.array: [description]
    """

    intrinsics_matrix = get_intrinsics_matrix(prim_path, depth_image.shape[1], depth_image.shape[0])
    
    fx = intrinsics_matrix[0][0]
    fy = intrinsics_matrix[1][1]
    cx = intrinsics_matrix[0][2]
    cy = intrinsics_matrix[1][2]
    height = depth_image.shape[0]
    width = depth_image.shape[1]
    input_x = np.arange(width)
    input_y = np.arange(height)
    input_x, input_y = np.meshgrid(input_x, input_y)
    input_x = input_x.flatten()
    input_y = input_y.flatten()
    input_z = depth_image.flatten()
    input_z[input_z > max_clip_depth] = 0
    output_x = (input_x * input_z - cx * input_z) / fx
    output_y = (input_y * input_z - cy * input_z) / fy
    raw_pc = np.stack([output_x, output_y, input_z], -1).reshape([height * width, 3])
    return raw_pc

def project_depth_to_worldspace(prim_path, depth_image, max_clip_depth):
    """Project depth image to world space

    Args:
        depth_image (np.array): Depth image buffer
        viewport_api (Any): Handle to viewport api
        max_clip_depth (float): Depth values larger than this will be clipped

    Returns:
        List[carb.Float3]: List of points from depth in world space
    """
    # import IPython
    # IPython.embed()
    start = time.time()
    stage = get_current_stage()
    # print("Time stage", time.time()-start)
    start = time.time()
    prim = stage.GetPrimAtPath(prim_path)
    # print("Time stage", time.time()-start)

    start = time.time()

    prim_tf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode())
    units_per_meter = 1.0 / UsdGeom.GetStageMetersPerUnit(stage)

    depth_data = depth_image * units_per_meter
    depth_data = -np.clip(depth_data, 0, max_clip_depth)
    # print("Time stage", time.time()-start)

    start = time.time()
    pc = backproject_depth(prim_path, depth_data, max_clip_depth)
    # print("Time stage", time.time()-start)

    # points = []
    # start = time.time()
    # pos = np.array(prim_tf.ExtractTranslation())
    # quat = prim_tf.ExtractRotationQuat()
    # rot = R.from_quat(np.concatenate([np.array(quat.imaginary), [quat.real] ])).as_matrix()
    # points = np.array(pc)
    # points[:,0] = - points[:,0]
    # new_points = np.matmul(points, rot)
    # points = new_points - pos
    # old_points = []
    new_pc = np.array(pc)
    new_pc[:,0] = - new_pc[:,0]
    transform = np.array(prim_tf)
    # import IPython
    # IPython.embed()
    # print("Transform", transform)
    points = np.concatenate([np.array(new_pc).T, np.ones((1,pc.shape[0]))])
    trans_points = transform.T@points
    trans_points = trans_points.T
    # old_points = []
    # for pts in pc:
    #     p = prim_tf.Transform(Gf.Vec3d(-pts[0], pts[1], pts[2]))
    #     old_points.append(carb.Float3(p[0], p[1], p[2]))
    # print("Time trans", time.time()-start)
    points = trans_points[:,:3]


    # points = np.array(old_points)
    return points


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Helper functions to project between pointcloud and depth images."""


import numpy as np
import torch
from typing import Optional, Sequence, Tuple, Union

import warp as wp

from omni.isaac.orbit.utils.array import convert_to_torch
from omni.isaac.orbit.utils.math import matrix_from_quat

__all__ = ["transform_points", "create_pointcloud_from_depth", "create_pointcloud_from_rgbd"]


"""
Depth <-> Pointcloud conversions.
"""


def transform_points(
    points: Union[np.ndarray, torch.Tensor, wp.array],
    position: Optional[Sequence[float]] = None,
    orientation: Optional[Sequence[float]] = None,
    device: Union[torch.device, str, None] = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Transform input points in a given frame to a target frame.

    This function uses torch operations to transform points from a source frame to a target frame. The
    transformation is defined by the position ``t`` and orientation ``R`` of the target frame in the source frame.

    .. math::
        p_{target} = R_{target} \times p_{source} + t_{target}

    If either the inputs `position` and `orientation` are :obj:`None`, the corresponding transformation is not applied.

    Args:
        points (Union[np.ndarray, torch.Tensor, wp.array]): An array of shape (N, 3) comprising of 3D points in source frame.
        position (Optional[Sequence[float]], optional): The position of source frame in target frame. Defaults to None.
        orientation (Optional[Sequence[float]], optional): The orientation ``(w, x, y, z)`` of source frame in target frame.
            Defaults to None.
        device (Optional[Union[torch.device, str]], optional): The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Returns:
        Union[np.ndarray, torch.Tensor]:
          A tensor of shape (N, 3) comprising of 3D points in target frame.
          If the input is a numpy array, the output is a numpy array. Otherwise, it is a torch tensor.
    """
    # check if numpy
    is_numpy = isinstance(points, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert to torch
    points = convert_to_torch(points, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = points.device
    # apply rotation
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # apply translation
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    # apply transformation
    points = _transform_points_jit(points, position, orientation)

    # return everything according to input type
    if is_numpy:
        return points.detach().cpu().numpy()
    else:
        return points


def create_pointcloud_from_depth(
    camera_prim_path,
    depth: Union[np.ndarray, torch.Tensor, wp.array],
    keep_invalid: bool = False,
    device: Optional[Union[torch.device, str]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    r"""Creates pointcloud from input depth image and camera intrinsic matrix.

    This function creates a pointcloud from a depth image and camera intrinsic matrix. The pointcloud is
    computed using the following equation:

    .. math::
        p_{camera} = K^{-1} \times [u, v, 1]^T \times d

    where :math:`K` is the camera intrinsic matrix, :math:`u` and :math:`v` are the pixel coordinates and
    :math:`d` is the depth value at the pixel.

    Additionally, the pointcloud can be transformed from the camera frame to a target frame by providing
    the position ``t`` and orientation ``R`` of the camera in the target frame:

    .. math::
        p_{target} = R_{target} \times p_{camera} + t_{target}

    Args:
        intrinsic_matrix (Union[np.ndarray, torch.Tensor, wp.array]): A (3, 3) array providing camera's calibration
            matrix.
        depth (Union[np.ndarray, torch.Tensor, wp.array]): An array of shape (H, W) with values encoding the depth
            measurement.
        keep_invalid (bool, optional): Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position (Optional[Sequence[float]], optional): The position of the camera in a target frame.
            Defaults to None.
        orientation (Optional[Sequence[float]], optional): The orientation ``(w, x, y, z)`` of the
            camera in a target frame. Defaults to None.
        device (Optional[Union[torch.device, str]], optional): The device for torch where the computation
            should be executed. Defaults to None, i.e. takes the device that matches the depth image.

    Raises:
        ValueError: When intrinsic matrix is not of shape (3, 3).
        ValueError: When depth image is not of shape (H, W) or (H, W, 1).

    Returns:
        Union[np.ndarray, torch.Tensor]:
          An array/tensor of shape (N, 3) comprising of 3D coordinates of points.
          The returned datatype is torch if input depth is of type torch.tensor or wp.array. Otherwise, a np.ndarray
          is returned.
    """
    # We use PyTorch here for matrix multiplication since it is compiled with Intel MKL while numpy
    # by default uses OpenBLAS. With PyTorch (CPU), we could process a depth image of size (480, 640)
    # in 0.0051 secs, while with numpy it took 0.0292 secs.

    stage = get_current_stage()
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    # Attention orbit is not using USdGeom
    prim_tf = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf = np.transpose(prim_tf)
    # Check state units assumed they are 1.0

    intrinsic_matrix = compute_intrinsic_matrix(camera_prim, depth)
    print(f'Inside of "create_pointcloud_from_depth"')
    print(f'Intrinsics matrix: {intrinsic_matrix}')
    #             intrinsics = np.array([[386.38769531,   0.        , 326.95498657],
    #                         [  0.        , 385.99816895, 237.9675293 ],
    #                         [  0.        ,   0.        ,   1.        ]])
    position = prim_tf[0:3, 3]
    cam_rotm = prim_tf[0:3, 0:3]
    cam_rotm[:,2] = - cam_rotm[:, 2]
    cam_rotm[:,1] = - cam_rotm[:,1]
    orientation = convert_quat(tf.Rotation.from_matrix(cam_rotm).as_quat(), "wxyz")

    # convert to numpy matrix
    is_numpy = isinstance(depth, np.ndarray)
    # decide device
    if device is None and is_numpy:
        device = torch.device("cpu")
    # convert depth to torch tensor
    depth = convert_to_torch(depth, dtype=torch.float32, device=device)
    # update the device with the device of the depth image
    # note: this is needed since warp does not provide the device directly
    device = depth.device
    # convert inputs to torch tensors
    intrinsic_matrix = convert_to_torch(intrinsic_matrix, dtype=torch.float32, device=device)
    if position is not None:
        position = convert_to_torch(position, dtype=torch.float32, device=device)
    if orientation is not None:
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)
    # compute pointcloud
    depth_cloud = _create_pointcloud_from_depth_jit(intrinsic_matrix, depth, keep_invalid, position, orientation)

    # return everything according to input type
    if is_numpy:
        return depth_cloud.detach().cpu().numpy()
    else:
        return depth_cloud



"""
Helper functions -- Internal
"""


@torch.jit.script
def _transform_points_jit(
    points: torch.Tensor,
    position: Optional[torch.Tensor] = None,
    orientation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Transform input points in a given frame to a target frame.

    Args:
        points (torch.Tensor): An array of shape (N, 3) comprising of 3D points in source frame.
        position (Optional[torch.Tensor], optional): The position of source frame in target frame. Defaults to None.
        orientation (Optional[torch.Tensor], optional): The orientation ``(w, x, y, z)`` of source frame in target frame.
            Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (N, 3) comprising of 3D points in target frame.
    """
    # -- apply rotation
    if orientation is not None:
        points = torch.matmul(matrix_from_quat(orientation), points.T).T
    # -- apply translation
    if position is not None:
        points += position

    return points


@torch.jit.script
def _create_pointcloud_from_depth_jit(
    intrinsic_matrix: torch.Tensor,
    depth: torch.Tensor,
    keep_invalid: bool = False,
    position: Optional[torch.Tensor] = None,
    orientation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Creates pointcloud from input depth image and camera intrinsic matrix.

    Args:
        intrinsic_matrix (torch.Tensor): A (3, 3) python tensor providing camera's calibration matrix.
        depth (torch.tensor): An tensor of shape (H, W) with values encoding the depth measurement.
        keep_invalid (bool, optional): Whether to keep invalid points in the cloud or not. Invalid points
            correspond to pixels with depth values 0.0 or NaN. Defaults to False.
        position (torch.Tensor, optional): The position of the camera in a target frame. Defaults to None.
        orientation (torch.Tensor, optional): The orientation ``(w, x, y, z)`` of the camera in a target frame.
            Defaults to None.

    Raises:
        ValueError: When intrinsic matrix is not of shape (3, 3).
        ValueError: When depth image is not of shape (H, W) or (H, W, 1).

    Returns:
        torch.Tensor: A tensor of shape (N, 3) comprising of 3D coordinates of points.

    """
    # squeeze out excess dimension
    if len(depth.shape) == 3:
        depth = depth.squeeze(dim=2)
    # check shape of inputs
    if intrinsic_matrix.shape != (3, 3):
        raise ValueError(f"Input intrinsic matrix of invalid shape: {intrinsic_matrix.shape} != (3, 3).")
    if len(depth.shape) != 2:
        raise ValueError(f"Input depth image not two-dimensional. Received shape: {depth.shape}.")
    # get image height and width
    im_height, im_width = depth.shape

    # convert image points into list of shape (3, H x W)
    indices_u = torch.arange(im_width, device=depth.device, dtype=depth.dtype)
    indices_v = torch.arange(im_height, device=depth.device, dtype=depth.dtype)
    img_indices = torch.stack(torch.meshgrid([indices_u, indices_v], indexing="ij"), dim=0).reshape(2, -1)
    pixels = torch.nn.functional.pad(img_indices, (0, 0, 0, 1), mode="constant", value=1.0)

    # convert into 3D points
    points = torch.matmul(torch.inverse(intrinsic_matrix), pixels)
    points = points / points[-1, :]
    points_xyz = points * depth.T.reshape(-1)
    # convert it to (H x W , 3)
    points_xyz = torch.transpose(points_xyz, dim0=0, dim1=1)
    # convert 3D points to world frame
    points_xyz = _transform_points_jit(points_xyz, position, orientation)

    # remove points that have invalid depth
    if not keep_invalid:
        pts_idx_to_keep = torch.all(torch.logical_and(~torch.isnan(points_xyz), ~torch.isinf(points_xyz)), dim=1)
        points_xyz = points_xyz[pts_idx_to_keep, ...]

    return points_xyz  # noqa: D504