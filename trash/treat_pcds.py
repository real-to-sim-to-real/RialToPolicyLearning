import numpy as np
import open3d as o3d
import time
from rialto.algo.spcnn import pad_points, preprocess_points_feats

folder_name="/scratch/marcel/data/demos/isaac-env/mugontable.usd_19_09_2023_19:42:35"
voxel_size=0.1
num_points = 4096
pad_points_b = False
downsample = True
crop_min = None
crop_max = None
only_state=False
pad_points_b = True
num_points = 8000
downsample = True
crop_min = -2
crop_max = 2
voxel_size = 0.01

import IPython
IPython.embed()
states = np.load(f"{folder_name}/demo_states.npy")

traj_length = np.array([states.shape[1] for i in range(states.shape[0])]).astype(np.int)
num_trajs = states.shape[0]
pcd_points = np.zeros((num_trajs, traj_length[0], num_points, 3))
pcd_feats = np.zeros((num_trajs, traj_length[0], num_points, 1))


for traj_idx in range(num_trajs):
    start = time.time()
    for time_idx in range(len(states[traj_idx])):
        filename = f"{folder_name}/traj_{traj_idx}/{time_idx}.pcd"
        pcd = o3d.io.read_point_cloud(filename)
        points, colors = preprocess_points_feats(np.asarray(pcd.points), np.asarray(pcd.colors), crop_min,crop_max,voxel_size, rgb_feats=False,num_points=num_points, add_padding=pad_points_b, downsample_points=downsample) #self.preprocess_pcd(pcd)
        pcd_points[traj_idx,time_idx] = points[0]
        pcd_feats[traj_idx,time_idx] = colors[0]
    print("done with traj", traj_idx, "of ", num_trajs, time.time() - start)

np.save(f"{folder_name}/all_points.npy", pcd_points)
np.save(f"{folder_name}/all_colors.npy", pcd_feats)

print("Loaded traj", traj_idx, time.time() - start)
# TODO: add num demos into account