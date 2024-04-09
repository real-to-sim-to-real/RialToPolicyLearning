import six
import trimesh
import numpy as np
import open3d as o3d
import time
import os
import yourdfpy
from transforms3d.euler import euler2quat
from transforms3d.quaternions import rotate_vector
import torch
from math import pi
try:
    from torchsparse import nn as spnn
    from torchsparse import SparseTensor
    from torchsparse.utils.collate import sparse_collate
    from torchsparse.utils.quantize import sparse_quantize
except ImportError as e:
    print(f'Could not import from torchsparse - sparse models unavailable ({e})')
import torchvision.transforms as transforms
from scipy.spatial.transform import Rotation as R

import torch
import wandb

def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new

def preprocess_points_feats_history(points, rgbd, crop_min, crop_max, voxel_size, rgb_feats=True, num_points=1024, add_padding=False, downsample_points=False):

    B, T, N, D_c = points.shape
    points = points.reshape(B*T, N, D_c)
    
    coords, feats = preprocess_points_feats(points, rgbd, crop_min, crop_max, voxel_size, rgb_feats, num_points, add_padding, downsample_points)

    TB, N_proc, _ = coords.shape
    coords = coords.reshape(B, T, N_proc, D_c)
    feats = feats.reshape(B, T, N_proc, 1)

    return coords, feats

def preprocess_points_feats(points, rgbd, crop_min, crop_max, voxel_size, rgb_feats=True, num_points=1024, add_padding=False, downsample_points=False):
    if rgb_feats and rgbd is not None:
        if len(points.shape) == 2:
            rgbd = rgbd.reshape(1, -1, 3)
        if len(points.shape) == 4:
            rgbd = np.stack(rgbd)
            rgbd = np.concatenate([rgbd[:,i,...,:3].reshape(rgbd.shape[0], -1, 3) for i in range(rgbd.shape[1])], axis=1)
    

    if len(points.shape) == 4:
        points = np.stack(points)
        points = np.concatenate((points[:,0], points[:,1]), axis=1)
    elif len(points.shape) == 2:
        points = points.reshape(1, -1, 3)
    
    coords = []
    feats = []
    
    for i, point in enumerate(points):
        # import IPython
        # IPython.embed()
        point = unpad_points(point)
        if not rgb_feats or rgbd is None:
            feat = np.ones((point.shape[0], 1))
        else:
            feat = feats[i]
        
        if rgb_feats:
            # normalize RGB
            feat = feats[i] / 255.

        coord, feat = quantize_points_feats(point, feat, crop_min=crop_min, crop_max=crop_max, voxel_size=voxel_size)
        
        # from helpers.pointclouds import print_pcd, points_to_pcd, visualize_pcd
        # pcd = points_to_pcd(coord, feat/255.)
        # print_pcd(pcd, filename="test.png")
        
        if coord.shape[0] < num_points:
            coord = pad_points(coord, num_points)
            feat = pad_points(feat, num_points)
        else:

            indices = np.random.choice(coord.shape[0], int(num_points), replace=False)
            coord = coord[indices]
            feat = feat[indices]

        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coord)
        # o3d.visualization.draw_geometries([pcd])
        coords.append(coord)
        feats.append(feat)

    coords = np.stack(coords)
    feats = np.stack(feats)

    return coords, feats




def preprocess_points_feats_pointnet(points, rgbd, crop_min, crop_max, voxel_size, rgb_feats=True, num_points=1024, add_padding=False, downsample_points=False):
    if rgb_feats and rgbd is not None:
        if len(points.shape) == 2:
            rgbd = rgbd.reshape(1, -1, 3)
        if len(points.shape) == 4:
            rgbd = np.stack(rgbd)
            rgbd = np.concatenate([rgbd[:,i,...,:3].reshape(rgbd.shape[0], -1, 3) for i in range(rgbd.shape[1])], axis=1)
    

    if len(points.shape) == 4:
        points = np.stack(points)
        points = np.concatenate((points[:,0], points[:,1]), axis=1)
    elif len(points.shape) == 2:
        points = points.reshape(1, -1, 3)
    

    lens = []
    new_points = []
    for i, point in enumerate(points):
        # import IPython
        # IPython.embed()
        point = unpad_points(point)
        if not rgb_feats or rgbd is None:
            feat = np.ones((point.shape[0], 1))
        else:
            feat = feats[i]
        
        if rgb_feats:
            # normalize RGB
            feat = feats[i] / 255.

        coord, feat = point, feat
        
        lens.append(len(coord))
        new_points.append(coord)
        # from helpers.pointclouds import print_pcd, points_to_pcd, visualize_pcd
        # pcd = points_to_pcd(coord, feat/255.)
        # print_pcd(pcd, filename="test.png")
        
    num_points = max(lens)
    coords = []
    feats = []
    for i, coord in enumerate(new_points):
        indices = np.random.choice(coord.shape[0], int(num_points), replace=False)
        coord = coord[indices]
        feat = feat[indices]

        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(coord)
        # o3d.visualization.draw_geometries([pcd])
        coords.append(coord)
        feats.append(feat)

    coords = np.stack(coords)
    feats = np.stack(feats)

    return coords, feats

def pad_points(points, M):
    N = points.shape[0]
    if M < N:
         return points
         raise ValueError(f"M ({M}) should be greater than or equal to N ({N})")
    return np.pad(points, ((0, M - N), (0, 0)), mode='constant', constant_values=-9999)

def unpad_points(padded_points):
    return padded_points[~np.all(padded_points < -9000, axis=1)]



def crop_points_feats(points, feats, crop_min, crop_max):
    valid = points[:,0] < crop_max[0]
    valid = np.logical_and(points[:,1] < crop_max[1], valid)
    valid = np.logical_and(points[:,2] < crop_max[2], valid)
    valid = np.logical_and(points[:,0] > crop_min[0], valid)
    valid = np.logical_and(points[:,1] > crop_min[1], valid)
    valid = np.logical_and(points[:,2] > crop_min[2], valid)

    new_points = points[valid]
    # new_feats = feats[valid_new]
    # import IPython
    # IPython.embed()
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])
    return new_points, feats

def quantize_points_feats_synthetic(points, feats, crop_min=None, crop_max=None, voxel_size=0.1):
    # crop points and feats
    # if crop_min is not None and crop_max is not None:
    #     # visualize_pointcloud(points, feats)
    #     points, feats = crop_points_feats(points, feats, crop_min, crop_max)
    #     # visualize_pointcloud(points, feats)

    coords = points

    # not sure if we can normalize here using the min as pointclouds vary
    # coords -= np.min(coords, axis=0, keepdims=True)
    # voxelize pointclouds
    coords, indices = sparse_quantize(coords,
                                        voxel_size,
                                        return_index=True)
    return coords, feats

def quantize_points_feats(points, feats, crop_min=None, crop_max=None, voxel_size=0.1):
    # crop points and feats
    # if crop_min is not None and crop_max is not None:
    #     # visualize_pointcloud(points, feats)
    #     points, feats = crop_points_feats(points, feats, crop_min, crop_max)
    #     # visualize_pointcloud(points, feats)

    coords = points

    # not sure if we can normalize here using the min as pointclouds vary
    # coords -= np.min(coords, axis=0, keepdims=True)
    # voxelize pointclouds
    coords, indices = sparse_quantize(coords,
                                        voxel_size,
                                        return_index=True)
    feats = feats[indices]
    return coords, feats

def coords_to_sparse(coords, feats, unpad=True, apply_augment=False, rgb_feats=False):

    sparse = []
    for coord, _ in zip(coords, feats):
        coord_pre = unpad_points(coord)
        # feat = unpad_points(feat)
        feat_pre = np.ones_like(coord_pre)[...,0].reshape(-1,1) # TODO: check if using rgb
        # import IPython
        # IPython.embed()
        # if apply_augment:
        #     coord, feat = apply_augment_points(coord, feat, proba_pcd=0.4, proba_rgb=0.3 if rgb_feats else 0.0)

        coord = torch.tensor(coord_pre, dtype=torch.int32).to("cuda")
        if not np.all(coord.detach().cpu().numpy() == coord_pre):
            import IPython
            IPython.embed()
        feat = torch.tensor(feat_pre, dtype=torch.float).to("cuda")
        tensor = SparseTensor(coords=coord, feats=feat)
        sparse.append(tensor)

    # combine and add batch dimension
    return sparse_collate(sparse)


def visualize_points(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def create_panda_urdf(cfg):

    class URDFWithCanonPCDS:
        def __init__(self, urdf):
            self.urdf = urdf
            self.canonical_link_pcds = None
        
        def __getattr__(self, name):
            return getattr(self.urdf, name)

        def build_canon_robot_pcds(self, n_pts_per_link):
            canonical_link_mesh_fnames = []
            canonical_link_mesh_meshes = []
            # canonical_link_pcds = []
            canonical_link_pcds = {}

            # for scene_link_name, scene_link_geom in urdf.scene.geometry.items():
                # link_fname = scene_link_geom.metadata['file_path']
            for link_name, link in self.link_map.items():
                link_fname = self._filename_handler(fname=link.visuals[0].geometry.mesh.filename)
                # link_fname = urdf._filename_handler(fname=link.collisions[0].geometry.mesh.filename)
                link_mesh = trimesh.load(link_fname, ignore_broken=True, force="mesh", skip_materials=True)
                link_pcd = link_mesh.sample(n_pts_per_link)

                canonical_link_mesh_fnames.append(link_fname)
                canonical_link_mesh_meshes.append(link_mesh)
                # canonical_link_pcds.append(link_pcd)
                canonical_link_pcds[link_name] = link_pcd
            
            self.canonical_link_pcds = canonical_link_pcds

    urdf = URDFWithCanonPCDS(yourdfpy.URDF.load("franka/new_franka/panda_with_gripper.urdf", build_collision_scene_graph=True, load_collision_meshes=True))
    urdf._create_scene(use_collision_geometry=False, force_mesh=True) # makes trimesh.Scene object
    urdf.build_canon_robot_pcds(n_pts_per_link=int(cfg['arm_num_points'] / len(urdf.cfg)))  # dict of point clouds - keys are link names, values are np.ndarrays (Nx3) in canonical pose, ready to be transformed by world link pose)

    return urdf

def get_arm_pcd(joints, urdf, arm_num_points):
    joints = joints[:-1].copy()
    joints[-1] = joints[-1]*0.04

    # urdf from: https://github.com/irom-lab/pybullet-panda/blob/main/data/franka/panda_arm.urdf
    urdf.update_cfg(joints) # This has to match the n-dof of actuated joints
    out_mesh = urdf.scene.dump(True)#.sum() # converts to trimesh.Trimesh object (full robot, not individual links)
    pts = out_mesh.sample(arm_num_points) # samples ponits on outer surface of mesh

    return pts 

def add_arm_pcd(points, joints, urdf, arm_num_points):
    # for i in range(points.shape[0]):
    arm_pcd = get_arm_pcd(joints, urdf, arm_num_points)
    new_points = np.concatenate([points, arm_pcd])

    return new_points

def randomize_joints(joints):
    print("TODO randomize joints")
    assert False
    return joints

def rotate_points(points):

    rot = euler2quat(0, pi, pi)
    new_points = []
    b = torch.tensor(points)
    a = torch.tensor(rot)
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:]
    t = xyz.cross(b, dim=-1) * 2
    new_points = (b + a[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

    return new_points.numpy()

def augment_pcds_with_history(batch_points, cfg):
    # import IPython
    # IPython.embed()
    # import copy
    # copy_batch_points = copy.deepcopy(batch_points)
    B, T, N, D_c = batch_points.shape
    batch_points = batch_points.reshape(T*B, N, D_c)
    
    coords = augment_pcds(batch_points, cfg)

    TB, N_aug, _ = coords.shape
    coords = coords.reshape(B, T, N_aug, D_c)

    #compare coords with original batch points
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(copy_batch_points[1,54])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(coords[1,54])
    # pcd2.colors = o3d.utility.Vector3dVector(np.zeros_like(coords[1,54]))
    # o3d.visualization.draw_geometries([pcd, pcd2])
    return coords

# def augment_pcds(batch_points, random_dropout=1, random_dropout_prob_min=0.1,random_dropout_prob_high=0.3, random_translation=1, random_translation_range=0.1, random_rot=1, random_rot_std=0.1, random_jitter=1, random_jitter_ratio=0.8, random_jitter_std=0.05):
def augment_pcds(batch_points, cfg):
    # A System for General In-Hand Object Re-Orientation. Chen et al. 2020
    B, N, C = batch_points.shape
    # Random dropout
    if np.random.uniform(0., 1.) < cfg['random_dropout']:
        # Sample different indices per batch
        ratio = np.random.uniform(cfg['random_dropout_prob_min'], cfg['random_dropout_prob_high']) 
        batch_points_down = []
        for i in range(B):
            indices = np.random.choice(N, int(ratio*N), replace=False)
            batch_points_down.append(np.delete(batch_points[i], indices, axis=0))
        batch_points =  np.array(batch_points_down)
    
        N = batch_points.shape[1]

    if np.random.uniform(0.,1.) < cfg['random_translation']:
        for i in range(B):
            translation = np.random.uniform(-cfg['random_translation_range'], cfg['random_translation_range']) # TODO: add ,3
            batch_points[i] +=  translation

    # # Random rotation
    # if np.random.uniform(0., 1.) < random_rot:
    #     # TODO: Look at the code I have on the sim to real transfer
    #     print("TODO rotation")
    #     # trans = np.random.uniform(-0.04, 0.04, size=3)
    #     # points += trans

    # Jitter points
    if np.random.uniform(0., 1.) < cfg['random_jitter']:
        # TODO add jitter
        for i in range(B):
            indices = np.random.choice(N, int(cfg['random_jitter_ratio']*N), replace=False)
            batch_points[i,indices] += np.clip(np.random.normal(0, cfg['random_jitter_std'], size=(len(indices), 3)), -0.015, 0.015)
        

    return batch_points


def downsample_internal(coord, feat, num_points):
    if (coord.shape[0]) < int(num_points):
        # import IPython
        # IPython.embed()
        # visualize_points(coord)
        # print("padding points")
        coord = pad_points(coord, num_points)
        
    indices = np.random.choice(coord.shape[0], int(num_points), replace=False)
    coord = coord[indices]
    # feat = feat[indices]
    return coord, feat

def downsample_internal_parallel(coord, feat, num_points):
    coords = np.zeros((coord.shape[0], num_points, 3))
    for i in range(coord.shape[0]):
        points_cropped, _ = crop_points_feats(coord[i], np.ones(len(points)), cfg['crop_min'], cfg['crop_max'])

        sample = np.random.choice(coord.shape[1], int(num_points), replace=False)
        coords[i] = coord[i, sample]
    # feat = feat[indices]
    return coord, feat

# def preprocess_pcd_synthetic(pcd, joint, urdf, cfg):
#     # pcd = pcd[0]
#     start = time.time()
#     pcd_processed_points = []
#     all_points, all_colors = pcd
#     start_par = time.time()
#     pcd_processed_points, _ = downsample_internal_parallel(all_points,  None, cfg['num_points'])
#     print("Downsample took", time.time() - start_par)
    
#     return pcd_processed_points

def preprocess_pcd_from_canon(pcd, joint_value, urdf, canonical_link_pcds, cfg):
    # pcd = pcd[0]
    start = time.time()
    pcd_processed_points = []
    pcd_processed_colors = []
    i = 0
    all_points, all_colors = pcd
    for points in all_points:
        now = time.time()
        # import IPython
        # IPython.embed()

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])
        
        points_cropped, _ = crop_points_feats(points, np.ones(len(points)), cfg['crop_min'], cfg['crop_max'])

        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_cropped)
        # o3d.visualization.draw_geometries([pcd])

        now = time.time()
        points, colors = downsample_internal(points_cropped,  np.ones(len(points)), cfg['num_points'])
        now = time.time()
        if not cfg["use_synthetic_pcd"]:
            # append all mimic joints in the update
            jnt = joint_value[i].detach().cpu().numpy()[:-1]
            jnt[-1] = jnt[-1]*0.02 + 0.02

            # urdf.update_cfg(jnt) # This has to match the n-dof of actuated joints
            configuration = jnt
            joint_cfg = []
            if isinstance(configuration, dict):
                for joint in configuration:
                    if isinstance(joint, six.string_types):
                        joint_cfg.append((urdf._joint_map[joint], configuration[joint]))
                    elif isinstance(joint, urdf.Joint):
                        # TODO: Joint is not hashable; so this branch will not succeed
                        joint_cfg.append((joint, configuration[joint]))
            elif isinstance(configuration, (list, tuple, np.ndarray)):
                if len(configuration) == len(urdf.robot.joints):
                    for joint, value in zip(urdf.robot.joints, configuration):
                        joint_cfg.append((joint, value))
                elif len(configuration) == urdf.num_actuated_joints:
                    for joint, value in zip(urdf._actuated_joints, configuration):
                        joint_cfg.append((joint, value))
                else:
                    raise ValueError(
                        f"Dimensionality of configuration ({len(configuration)}) doesn't match number of all ({len(urdf.robot.joints)}) or actuated joints ({urdf.num_actuated_joints})."
                    )
            else:
                raise TypeError("Invalid type for configuration")

            tf_pcd_list = []
            for j, q in joint_cfg + [
                (j, 0.0) for j in urdf.robot.joints if j.mimic is not None
            ]:
                matrix, joint_q = urdf._forward_kinematics_joint(j, q=q)

                # update internal configuration vector - only consider actuated joints
                if j.name in urdf.actuated_joint_names:
                    urdf._cfg[
                        urdf.actuated_dof_indices[urdf.actuated_joint_names.index(j.name)]
                    ] = joint_q
                
                # print(f'Matrix: {matrix}, q: {q}')

                # update internal configuration vector - only consider actuated joints
                if j.name in urdf.actuated_joint_names:
                    urdf._cfg[
                        urdf.actuated_dof_indices[urdf.actuated_joint_names.index(j.name)]
                    ] = joint_q

                if urdf._scene is not None:
                    urdf._scene.graph.update(
                        frame_from=j.parent, frame_to=j.child, matrix=matrix
                    )
                if urdf._scene_collision is not None:
                    urdf._scene_collision.graph.update(
                        frame_from=j.parent, frame_to=j.child, matrix=matrix
                    )
                
                world_pose = urdf.scene.graph.get(frame_to=j.child, frame_from=urdf.scene.graph.base_frame)[0]
                if j.child == "panda_rightfinger":
                    # flip the last finger around
                    zrot = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                    ztf = np.eye(4); ztf[:-1, :-1] = zrot
                    world_pose = np.matmul(world_pose, ztf)
                # print(f'World pose: {world_pose}')

                tf_pcd = transform_pcd(canonical_link_pcds[j.child], world_pose)
                tf_pcd_list.append(tf_pcd)
            
            arm_pcd = np.concatenate(tf_pcd_list, axis=0)

            points = np.concatenate([points, arm_pcd])

        pcd_processed_points.append(pad_points(points, cfg['num_points']))
        # pcd_processed_colors.append(colors)
        i+= 1

        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])

    pcd_proc_points = np.array(pcd_processed_points)
    print("preprocessing time", time.time() - start)

    # pcd_proc_colors = np.array(pcd_processed_colors)
    return pcd_proc_points


def preprocess_pcd(pcd, joint, urdf, cfg):
    # pcd = pcd[0]
    start = time.time()
    pcd_processed_points = []
    pcd_processed_colors = []
    i = 0
    all_points, all_colors = pcd
    for points in all_points:
        points_cropped, _ = crop_points_feats(points, np.ones(len(points)), cfg['crop_min'], cfg['crop_max'])
        points, colors = downsample_internal(points_cropped,  np.ones(len(points)), cfg['num_points'])
        if not cfg["use_synthetic_pcd"]:
            points = add_arm_pcd(points, joint[i].detach().cpu().numpy(), urdf, cfg['arm_num_points'])

        pcd_processed_points.append(pad_points(points, cfg['num_points']))
        # pcd_processed_colors.append(colors)
        i+= 1

        # import IPython
        # IPython.embed()
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.visualization.draw_geometries([pcd])

    pcd_proc_points = np.array(pcd_processed_points)
    print("preprocessing time", time.time() - start)

    # pcd_proc_colors = np.array(pcd_processed_colors)
    return pcd_proc_points

def unpad_and_downsample(points, cfg):
    idx = np.where(points < -9000)
    if len(idx[0])==0 and points.shape[1] < cfg["num_points"]:
        return points

    if len(idx[0])!=0:
        min_num_points = cfg["num_points"]
        all_unpadded = []
        for i in range(points.shape[0]):
            unpadded_point = unpad_points(points[i])
            min_num_points = min(min_num_points, len(unpadded_point))
            all_unpadded.append(unpadded_point)
    else:
        min_num_points = cfg["num_points"]
        all_unpadded = points

    downsampled_points = []
    for i in range(points.shape[0]):
        downsampled_points.append(downsample_internal(all_unpadded[i], None, min_num_points)[0])

    return np.array(downsampled_points)
    
def random_yaw_orientation_with_bounds_object(lower, upper, num, device="cuda:0") -> torch.Tensor:
    from omni.isaac.orbit.utils.math import quat_inv, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz


    """Returns sampled rotation around z-axis.

    Args:
        num (int): The number of rotations to sample.
        device (str): Device to create tensor on.

    Returns:
        torch.Tensor: Sampled quaternion (w, x, y, z).
    """
    lower = torch.tensor(lower).to(device)
    upper = torch.tensor(upper).to(device)
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = (upper - lower) * torch.rand(num, dtype=torch.float, device=device) + lower

    return quat_from_euler_xyz(roll, pitch, yaw)

def add_synthetic_distractors(pcd_processed_points, cfg, device="cuda:0"):
    from omni.isaac.orbit.utils.math import quat_inv, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
    from omni.isaac.orbit.utils.math import (
        quat_apply,
        quat_mul,
        matrix_from_quat
    )

    B = pcd_processed_points.shape[0]
    mesh_num_points = 300
    for mesh_name in cfg["mesh_names"]:
        mesh = np.load(f'{cfg["foldermeshname"]}/{mesh_name}.npy')
        # different numbers per pointcloud
        # rotate mesh
        # pos mesh
                
        lower = torch.tensor(cfg["distractor_pos_min_bound"]).to(device)
        upper = torch.tensor(cfg["distractor_pos_max_bound"]).to(device)
        # object_name = self.cfg.randomization.randomize_object_name

        pos = sample_uniform(
                lower, upper, (B, 3), device=device
            )

        rot = random_yaw_orientation_with_bounds_object(cfg["distractor_ori_min_bound"], cfg["distractor_ori_max_bound"], B)
        choice = np.random.choice(len(mesh), mesh_num_points , replace=False)
        mesh = mesh[choice]
        # mesh = [ mesh for i in range(100)]
        final_shape = (B, *mesh.shape)
        mesh = np.broadcast_to(mesh, final_shape)

        rot = np.broadcast_to(rot.cpu().numpy(), (mesh_num_points,B,4)).transpose(1,0,2)
        rotated_points = quat_apply( torch.tensor(rot).to("cuda").float(), torch.tensor(mesh).to("cuda").float())
        trans_points = rotated_points + pos.unsqueeze(1)
        trans_points = trans_points.detach().cpu().numpy()
        pcd_processed_points  = np.concatenate([pcd_processed_points, trans_points],axis=1)

    return pcd_processed_points

def postprocess_pcd(pcd_processed_points, cfg):
    start = time.time()
    if cfg["add_synthetic_distractors"]:
        pcd_processed_points = add_synthetic_distractors(pcd_processed_points, cfg)
    

    pcd_proc_points_aug = augment_pcds(pcd_processed_points, cfg)
    
    if cfg["pcd_encoder_type"] == "dense_conv":
        pcd_processed_points_full = unpad_and_downsample(pcd_proc_points_aug, cfg)
    else:
        pcd_processed_points_full, _ = preprocess_points_feats(pcd_proc_points_aug, None, cfg['crop_min'],cfg['crop_max'],cfg['voxel_size'], rgb_feats=False,num_points=cfg['num_points'], add_padding=cfg['pad_points'], downsample_points=cfg['downsample']) #self.preprocess_pcd(pcd)
    print("postprocessing time", time.time() - start)
    feats = np.ones((*pcd_processed_points_full.shape[:-1], 1))
    return pcd_processed_points_full, feats

# def postprocess_pcd_synthetic(pcd_processed_points, cfg):
#     start_par = time.time()

#     pcd_proc_points_aug = augment_pcds(pcd_processed_points, cfg)
#     print("augment took", time.time() - start_par)
#     start_par = time.time()

#     if cfg["pcd_encoder_type"] == "conv3d":
#         pcd_processed_points_full = pcd_proc_points_aug
#     else:
#         pcd_processed_points_full, _ = preprocess_points_feats_synthetic(pcd_proc_points_aug, None, cfg['crop_min'],cfg['crop_max'],cfg['voxel_size'], rgb_feats=False,num_points=cfg['num_points'], add_padding=cfg['pad_points'], downsample_points=cfg['downsample'], cfg=cfg) #self.preprocess_pcd(pcd)
#     print("preprocess points feats", time.time() - start_par)
#     feats = np.ones((*pcd_processed_points_full.shape[:-1], 1))

#     return pcd_processed_points_full, feats

def create_video(images, video_filename):
        images = np.array(images).astype(np.uint8)

        images = images.transpose(0,3,1,2)
        
        wandb.log({video_filename:wandb.Video(images, fps=10)})

def visualize_trajectory(images, success, name="", max=1):
    print("Visualize trajectory")
    start = time.time()
    if np.sum(success)>0:
        images_success = np.concatenate(images[success][:max], axis=1)
        create_video(images_success, "success"+name)

    if np.sum(success)< success.shape[0]:
        images_failed = np.concatenate(images[np.logical_not(success)][:max], axis=1)
        create_video(images_failed, "failed"+name)

    print("Visualize_trajectory took ", time.time() - start)

def create_env(cfg_eval, display=False, seed=0, env_name="isaac-env"):
    import gym
    import numpy as np
    from omni.isaac.kit import SimulationApp
    
    import rlkit.torch.pytorch_util as ptu
    ptu.set_gpu_mode(True, 0)

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu
    config = {"headless": not display}

    # load cheaper kit config in headless
    # launch the simulator
    simulation_app = SimulationApp(config)
    # Envs

    from rialto import envs
    from rialto.envs.env_utils import DiscretizedActionEnv

    # Algo
    from rialto.algo import buffer, variants, networks

    ptu.set_gpu(0)

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = envs.create_env(env_name, continuous_action_space=cfg_eval["continuous_action_space"],randomize_object_name=cfg_eval["randomize_object_name"], randomize_action_mag=cfg_eval["randomize_action_mag"], display=display, render_images=cfg_eval["render_images"], img_shape=(cfg_eval["img_width"],cfg_eval["img_height"]),usd_path=cfg_eval["usd_path"], usd_name=cfg_eval['usd_name'], num_envs=cfg_eval["num_envs"], sensors=cfg_eval["sensors"], num_cameras=cfg_eval["num_cameras"], randomize_pos=cfg_eval["randomize_pos"], randomize_rot=cfg_eval["randomize_rot"], cfg=cfg_eval)

    env_params = envs.get_env_params(env_name)
    env_params['continuous_action_space'] = cfg_eval['continuous_action_space']
    env = variants.get_env(env, env_params)

    env.reset()
    env.step(np.zeros(cfg_eval["num_envs"]).astype(np.int32))
    env.render_image(cfg_eval["sensors"])
    env.reset()

    return env, simulation_app

def create_state_policy(cfg, env, model_name=None, run_path=None):
    from rialto import envs
    from rialto.algo import variants

    env_params = envs.get_env_params(cfg["env_name"])
    env_params['network_layers']=cfg["expert_network"]
    env_params['use_horizon'] = False
    env_params['fourier'] = False
    env_params['fourier_goal_selector'] = False
    env_params['normalize']=False
    env_params['env_name'] = cfg["env_name"]
    env_params['continuous_action_space'] = cfg['continuous_action_space']
    policy = variants.get_policy(env, env_params)

    gpu = cfg["gpu"]
    
    if model_name:
        if cfg["from_ppo"]:
            from launch_ppo import WrappedIsaacEnv, PPO
            env2 = WrappedIsaacEnv(env, max_path_length=cfg["max_path_length"], from_vision=False, cfg=cfg)

            # Store policy in our format so it's easy to use after
            mapping = {
                "mlp_extractor.shared_net.0.weight": "net.net.network.0.weight",
                "mlp_extractor.shared_net.0.bias":"net.net.network.0.bias",
                "mlp_extractor.shared_net.2.weight": "net.net.network.2.weight",
                "mlp_extractor.shared_net.2.bias": "net.net.network.2.bias",
                "action_net.weight": "net.net.network.4.weight",
                "action_net.bias": "net.net.network.4.bias",
            }
            # if weights is None:
            new_weights = {}
            # else:
            #     new_weights = weights.copy()
            from torch import nn
            policy_kwargs = dict()
            policy_kwargs['net_arch'] = [int(l) for l in cfg["expert_network"].split(",")]
            policy_kwargs['activation_fn'] = nn.ReLU
            new_model = PPO("MlpPolicy", env2, verbose=2, ent_coef = 1e-2, n_steps=1, batch_size=10, policy_kwargs=policy_kwargs, device="cuda", from_vision=False)
            model_old = wandb.restore(f"checkpoints/{model_name}.zip", run_path=run_path)
            new_model = new_model.load(model_old.name)
            policy_weights = new_model.policy.state_dict()
       
            for key in mapping:
                new_weights[mapping[key]] = policy_weights[key]
            
            state_dict = new_weights
        else:
            expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
            state_dict = torch.load(expert_policy.name, map_location=f"cuda:{gpu}")
            if "net.net.obs_f.bias" in state_dict:
                del state_dict["net.net.obs_f.bias"]
                del state_dict["net.net.obs_f.weight"]
            

        policy.load_state_dict(state_dict)

    policy = policy.to(f"cuda:{gpu}")

    return policy


# from huge.algo.spcnn import SparseConvPolicy, SparseRNNConvPolicy
from rialto.algo.pointnet import PointNetPolicy
from rialto.algo.dense_cnn_model import DenseConvPolicy

def create_pcd_policy(cfg, env=None, model_name=None, run_path=None):
    layers = cfg["layers"]
    if isinstance(layers, str):
        layers = [int(x) for x in layers.split(",")]

    if cfg["pcd_encoder_type"] == "pointnet":
        policy_distill = PointNetPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            # obs_size=env.observation_space.shape[0],
            act_size=cfg['act_dim'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            layers=layers,
            dropout=cfg["dropout"],
        )
    elif cfg["pcd_encoder_type"] == "dense_conv":
        plane_type = None
        if cfg["plane_type"] == "grid":
            plane_type = ['grid']
        elif cfg["plane_type"] == "grid_and_planes":
            plane_type = ['xz', 'xy', 'yz','grid']
        elif cfg["plane_type"] == "planes":
            plane_type = ['xz', 'xy', 'yz']
        else:
            print("ERROR: wrong plane_type passed as parameter")
            assert False

        scene_encoder_kwargs = {
            'local_coord': True,
            # 'encoder': pointnet_local_pool,
            'c_dim': 32,
            # 'encoder_kwargs':
            'hidden_dim': 32,
            # 'plane_type': ['xz', 'xy', 'yz', 'grid'],
            'plane_type': plane_type,
            # 'grid_resolution': 32,
            'plane_resolution': cfg["unet3d"]["plane_resolution"],
            'unet3d': True,
            'unet3d_kwargs': {
                'num_levels': cfg["unet3d"]["num_levels"], #3,
                'f_maps': cfg["unet3d"]["f_maps"], # 32,
                'in_channels': cfg["unet3d"]["in_channels"], #32,
                'out_channels': cfg["unet3d"]["out_channels"], #64, #32,
                'plane_resolution': cfg["unet3d"]["plane_resolution"], #128,
            },
            'unet': False,
            # 'unet_kwargs': {
            #     'depth': 5,
            #     'merge_mode': 'concat',
            #     'start_filts': 32
            # }
        }
        policy_distill = DenseConvPolicy(
            in_channels= 3 if cfg['rgb_feats'] else 1,
            # obs_size=env.observation_space.shape[0],
            act_size=cfg['act_dim'],
            rgb_feats=cfg['rgb_feats'],
            pad_points=cfg['pad_points'],
            use_state=cfg['use_state'],
            layers=layers,
            encoder_type=cfg["pcd_encoder_type"],
            dropout=cfg["dropout"],
            pool=cfg["pool"],
            pcd_scene_scale=cfg["pcd_scene_scale"],
            pcd_normalization=cfg["pcd_normalization"],
            scene_encoder_kwargs=scene_encoder_kwargs
        )

    else:
        if cfg["rnn"]:
            policy_distill = SparseRNNConvPolicy(
                in_channels= 3 if cfg['rgb_feats'] else 1,
                # obs_size=env.observation_space.shape[0],
                act_size=cfg['act_dim'],
                hidden_size=cfg['hidden_size'],
                rgb_feats=cfg['rgb_feats'],
                pad_points=cfg['pad_points'],
                gru=cfg['gru'],
                use_state=cfg['use_state'],
                layers=layers,
                encoder_type=cfg["pcd_encoder_type"],
                dropout=cfg["dropout"],
                pool=cfg["pool"],
            )
        else:
            policy_distill = SparseConvPolicy(
                in_channels= 3 if cfg['rgb_feats'] else 1,
                # obs_size=env.observation_space.shape[0],
                act_size=cfg['act_dim'],
                rgb_feats=cfg['rgb_feats'],
                pad_points=cfg['pad_points'],
                use_state=cfg['use_state'],
                layers=layers,
                encoder_type=cfg["pcd_encoder_type"],
                dropout=cfg["dropout"],
                pool=cfg["pool"],
            )

    gpu = cfg["gpu"]
    device = f"cuda:{gpu}"

    if model_name:
        if model_name == str(-1):
            policy_distill.load_state_dict(torch.load(f"{cfg['datafolder']}/policy_distill.pt", map_location=device))
        else:
            if cfg["model_from_disk"]:
                expert_policy = torch.load(f"checkpoints/{model_name}.pt", map_location=device)
                policy_distill.load_state_dict(expert_policy)
            else:
                expert_policy = wandb.restore(f"checkpoints/{model_name}.pt", run_path=run_path)
                policy_distill.load_state_dict(torch.load(expert_policy.name, map_location=device))
    
    policy_distill.to(device)

    return policy_distill

def extract_state(state):
    return state[:, -9:] # tool x2 rotation x4 position x3

@torch.no_grad()
def rollout_policy(env, policy, urdf, cfg, render=True, from_state=True, expert_from_state=True, expert_policy=None, visualize_traj=False, sampling_expert=1):
    policy.eval()
    if expert_policy:
        expert_policy.eval()

    actions = []
    states = []
    joints = []
    cont_actions = []
    all_pcds_points = []
    all_pcds_points_full = []
    all_pcds_colors_full = []
    expert_actions = []
    images = []

    state = env.reset()
    joint = env.base_env._env.get_robot_joints()
    start_demo =time.time()
    resize_transform = transforms.Resize((64, 64))
    device = f"cuda:{cfg['gpu']}"
    debug = False

    for t in range(cfg['max_path_length']):

        if render or visualize_traj:
            start = time.time()
            img, pcd = env.render_image(cfg["sensors"])
            print("Rendering pcd image", time.time()-start)
            # if cfg["use_synthetic_pcd"]:
            #     pcd_processed_points = preprocess_pcd_synthetic(pcd, joint, urdf, cfg)
            #     pcd_processed_points_full, pcd_processed_colors_full = postprocess_pcd_synthetic(pcd_processed_points, cfg) 
            # else:
            if cfg["presample_arm_pcd"]:
                # this is faster (doesn't re-sample the points from the mesh each time)
                pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
            else:
                pcd_processed_points = preprocess_pcd(pcd, joint, urdf, cfg)
            

            # import open3d as o3d
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points[0]))
            # o3d.visualization.draw_geometries([pcd])

            pcd_processed_points_full, pcd_processed_colors_full = postprocess_pcd(pcd_processed_points, cfg) 


            if debug:
                import IPython
                IPython.embed()
                real_demos_actions =  np.load("demos/booknshelveteleopbig/actions_0_0.npy")
                import open3d as o3d
                demo_pcd = np.load("demos/booknshelveteleopbig/pcd_points_0_0.npy")
                pcd_demo = o3d.geometry.PointCloud()
                pcd_demo.points = o3d.utility.Vector3dVector(demo_pcd[0,t])
                pcd_demo.colors = o3d.utility.Vector3dVector(np.zeros_like(demo_pcd[0,0]))
                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points[0]))
                o3d.visualization.draw_geometries([pcd_demo, pcd_current])


                real_pcd = np.load("real_pcd_unpad.npy")
                pcd_demo = o3d.geometry.PointCloud()
                pcd_demo.points = o3d.utility.Vector3dVector(real_pcd)
                pcd_demo.colors = o3d.utility.Vector3dVector(np.zeros_like(real_pcd))
                pcd_current = o3d.geometry.PointCloud()
                pcd_current.points = o3d.utility.Vector3dVector(unpad_points(pcd_processed_points_full[0]))
                o3d.visualization.draw_geometries([pcd_demo, pcd_current])

                # debug to show both ways of obtaining arm pcds on top of each other
                # import open3d as o3d
                # pcd_processed_points = preprocess_pcd_from_canon(pcd, joint, urdf, urdf.canonical_link_pcds, cfg)
                # pcd_processed_points2 = preprocess_pcd(pcd, joint, urdf, cfg)
                # pcdv1 = o3d.geometry.PointCloud()
                # pcdv1.points = o3d.utility.Vector3dVector(pcd_processed_points[0])
                # pcdv2 = o3d.geometry.PointCloud()
                # pcdv2.points = o3d.utility.Vector3dVector(pcd_processed_points2[0])
                # pcdv1.paint_uniform_color([1.0, 0.0, 0.0])
                # pcdv2.paint_uniform_color([0.0, 1.0, 0.0])
                # o3d.visualization.draw_geometries([pcdv1, pcdv2])

        observation = env.observation(state)

        if from_state:
            action = policy.act_vectorized(observation, observation)
        else:
            if cfg["rnn"]:
                with torch.no_grad():
                    assert False # TODO change joints
                    pcd_processed_points_full_par, pcd_processed_colors_full_par, joint_par = np.expand_dims(pcd_processed_points_full, axis=1), np.expand_dims(pcd_processed_colors_full, axis=1), joint.unsqueeze(1)
                    action = policy((pcd_processed_points_full_par, pcd_processed_colors_full_par, joint_par), init_belief=(t==0)).detach().squeeze()    

            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(extract_state(observation)).to(device)
                    action = policy((pcd_processed_points_full, pcd_processed_colors_full, obs_tensor))


            if cfg["sample_action"]:
                action = torch.distributions.Categorical(torch.softmax(action, axis=1)).sample().cpu().numpy()
            else:
                action = action.argmax(dim=1).cpu().numpy()


        if debug:
            action = np.array([real_demos_actions[0,t] for i in range(state.shape[0])])
            print("action obs", action, observation)

        if expert_policy:
            if expert_from_state:
                expert_action = expert_policy.act_vectorized(observation, observation)
            else:
                obs_tensor = torch.tensor(extract_state(observation)).to(device)
                expert_action = expert_policy((pcd_processed_points_full, pcd_processed_colors_full, obs_tensor))
                
                if cfg["sample_action"]:
                    expert_action = torch.distributions.Categorical(torch.softmax(expert_action, axis=1)).sample().cpu().numpy()
                else:
                    expert_action = expert_action.argmax(dim=1).cpu().numpy()

            expert_actions.append(expert_action)

        if visualize_traj:
            all_pcds_points.append(pcd_processed_points)
            all_pcds_points_full.append(pcd_processed_points_full)
            # all_pcds_colors_full.append(pcd_processed_colors_full)
            
            # Convert the numpy array to a PyTorch tensor
            tensor = torch.tensor(img).permute(0, 3, 1, 2)  # Change shape to (N, 3, width, height)

            img = resize_transform(tensor).permute(0,2,3,1).cpu().numpy()

            images.append(img) 

        if expert_policy and np.random.random() < sampling_expert :
            action = expert_action

        actions.append(action)
        states.append(state)
        joints.append(joint.detach().cpu().numpy())

        state, _, done , info = env.step(action)
        joint = info["robot_joints"]
        cont_action = info["cont_action"].detach().cpu().numpy()
        cont_actions.append(cont_action)

    success = env.base_env._env.get_success().detach().cpu().numpy()
    
    print(f"Trajectory took {time.time() - start_demo}")

    actions = np.array(actions).transpose(1,0)
    cont_actions = np.array(cont_actions).transpose(1,0,2)
    states = env.observation(np.array(states).transpose(1,0,2))
    joints = np.array(joints).transpose(1,0,2)

    if expert_policy:
        expert_actions = np.array(expert_actions).transpose(1,0)
    else:
        expert_actions = None

    if visualize_traj:
        print("Creating arrays of pcd")
        all_pcd_points = np.array(all_pcds_points).transpose(1,0,2,3)
        all_pcd_colors = np.ones_like(all_pcd_points)
        all_pcd_colors = all_pcd_colors[...,0]
        all_pcd_colors = all_pcd_colors[...,None]
        all_pcd_points_full = np.array([])#np.array(all_pcds_points_full).transpose(1,0,2,3)
        all_pcd_colors_full = np.array([])#np.array(all_pcds_colors_full).transpose(1,0,2,3)
        images = np.array(images).transpose(1,0,2,3,4)
    else:
        all_pcd_points = None
        all_pcd_colors = None
        all_pcd_points_full = None
        all_pcd_colors_full = None

    wandb.log({"Success": np.mean(success), "Time/Trajectory": time.time() - start_demo})
    return actions, cont_actions, states, joints, all_pcd_points_full, all_pcd_colors_full, all_pcd_points, all_pcd_colors, images, expert_actions, success
