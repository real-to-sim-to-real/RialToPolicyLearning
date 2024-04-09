import numpy as np
import open3d as o3d
from rialto.algo.spcnn import pad_points, preprocess_points_feats
import time
import os

# In this 
class Buffer:
    def __init__(self, 
                 folder_name="demos/isaac-env/kitchenv3close.usd_30_08_2023_18:47:50", 
                 voxel_size=0.1,
                 num_points = 4096,
                 pad_points = False,
                 downsample = True,
                 crop_min = None,
                 crop_max = None,
                 only_state=False,
                 num_demos=None, 
                 validation_split=0.2
                 ):
        self.only_state = only_state
        self.num_demos = num_demos
        self.folder_name = folder_name
        self.voxel_size = voxel_size
        self.downsample = downsample
        self.crop_min = crop_min
        self.crop_max = crop_max

        self.num_points = num_points
        self.pad_points = pad_points
    
        # load all actions
        print("foler name", self.folder_name)
        env_folder_name = folder_name.split("/")[-1]
        if self.only_state:
            if os.path.exists(f"demos/{env_folder_name}/demo_{0}_states.npy"):
                state = np.load(f"demos/{env_folder_name}/demo_{0}_states.npy")
                obs_dim = len(state[0])//3
                max_path_length = 50
                self.actions = np.zeros((num_demos, max_path_length))
                self.states = np.zeros((num_demos, max_path_length, obs_dim))
                self.traj_length = np.zeros(num_demos)
                for i in range(self.num_demos):
                    states = np.load(f"demos/{env_folder_name}/demo_{i}_states.npy")
                    traj_length = len(states)
                    self.actions[i,:traj_length]=np.load(f"demos/{env_folder_name}/demo_{i}_actions.npy").astype(np.int)
                    self.states[i, :traj_length]= states[:,:len(states[0])//3]
                    self.traj_length[i]= traj_length
                self.num_trajs = num_demos
            else:
                self.actions = np.load(f"{self.folder_name}/demo_actions.npy").astype(np.int)
                self.states = np.load(f"{self.folder_name}/demo_states.npy")
                self.joints = np.load(f"{self.folder_name}/demo_joints.npy")
                self.traj_length = np.array([self.states.shape[1] for i in range(self.states.shape[0])]).astype(np.int)
                self.num_trajs = self.states.shape[0]

        else:
            self.actions = np.load(f"{self.folder_name}/demo_actions.npy").astype(np.int)
            self.states = np.load(f"{self.folder_name}/demo_states.npy")
            self.joints = np.load(f"{self.folder_name}/demo_joints.npy")
            
            self.traj_length = np.array([self.states.shape[1] for i in range(self.states.shape[0])]).astype(np.int)
            self.num_trajs = self.states.shape[0]
            if self.num_demos is not None:
                self.num_trajs = self.num_demos
                
            # if os.path.exists(f"{self.folder_name}/all_points.npy"):
            #     print("ATTENTION loading files")
            #     self.pcd_points = np.load(f"{self.folder_name}/all_points.npy")
            #     self.pcd_feats = np.load(f"{self.folder_name}/all_colors.npy")


            self.pcd_points = np.zeros((self.num_demos, self.traj_length[0], num_points, 3))
            self.pcd_feats = np.zeros((self.num_demos, self.traj_length[0], num_points, 1))

            
            for traj_idx in range(self.num_trajs):
                start = time.time()
                for time_idx in range(len(self.states[traj_idx])):
                    # filename = f"{self.folder_name}/traj_{traj_idx}/{time_idx}.pcd"
                    # pcd = o3d.io.read_point_cloud(filename)
                    # points, colors = preprocess_points_feats(np.asarray(pcd.points), np.asarray(pcd.colors), self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample) #self.preprocess_pcd(pcd)
                    points = np.load(f"{self.folder_name}/traj_{traj_idx}/{time_idx}_points.npy")
                    colors = np.load(f"{self.folder_name}/traj_{traj_idx}/{time_idx}_feats.npy")
                    points, colors = preprocess_points_feats(points, colors, self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample) #self.preprocess_pcd(pcd)
                    
                    self.pcd_points[traj_idx,time_idx] = points[0]
                    self.pcd_feats[traj_idx,time_idx] = colors[0]
                print("Loaded traj", traj_idx, time.time() - start)
            np.save(f"{self.folder_name}/all_points.npy", self.pcd_points)
            np.save(f"{self.folder_name}/all_colors.npy", self.pcd_feats)
        # TODO: add num demos into account
        self.traj_length = self.traj_length.astype(np.int)
        self.actions = self.actions.astype(np.int)
        self.obs_size = len(self.states[0][0])
        split = np.random.random(self.num_trajs)< validation_split
        self.validation_idx = np.where(split)
        self.all_idx = np.where(np.logical_not(split))

        self.action_joints = self.joints[:, :-1]

    def preprocess_pcd(self, pcd):
        downpcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if self.pad_points:
            points = pad_points(np.asarray(downpcd.points), self.num_points)
            colors = pad_points(np.asarray(downpcd.colors), self.num_points)
        else:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
            colors = colors[indices]

        # new_pcd = o3d.geometry.PointCloud()
        # new_pcd.points = points
        # new_pcd.colors = colors

        return points, colors
        
    def sample(self, batch_size, validation=False):
        if validation:
            traj_idxs = np.random.choice(self.validation_idx[0], batch_size)
        else:
            traj_idxs = np.random.choice(self.all_idx[0], batch_size)

        traj_length = self.traj_length[traj_idxs]
        time_idxs = np.array([np.random.choice(traj_length[i],1)[0] for i in range(batch_size)])
        batch_actions = self.actions[traj_idxs, time_idxs]
        if self.only_state:
            batch_states = self.states[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
        else:
            batch_states = self.joints[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
        batch_points = []
        batch_feats = []
        start = time.time()
        if not self.only_state:
            batch_points = self.pcd_points[traj_idxs,time_idxs]
            batch_feats = self.pcd_feats[traj_idxs,time_idxs]


        print(f"Buffer sampling", time.time() - start)

        return batch_points, batch_feats, batch_states, batch_actions