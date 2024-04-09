import numpy as np
import open3d as o3d
import time
import os
from utils import create_panda_urdf, downsample_internal, postprocess_pcd,preprocess_points_feats_pointnet, crop_points_feats, add_arm_pcd, augment_pcds, augment_pcds_with_history, preprocess_points_feats_history, extract_state
# Pipeline:
# Fetch all pcds from memory
# downsample to fit the array
# sample batch
# add randomization to the pcd
# voxelize
# remove duplicates from voxelization
# convert to sparse tensors
# forward pass on sparse tensors
import wandb
import torch
# In this 
# class Buffer:
#     def __init__(self, 
#                  folder_name="demos/isaac-env/kitchenv3close.usd_30_08_2023_18:47:50", 
#                  voxel_size=0.1,
#                  num_points = 4096,
#                  pad_points = False,
#                  downsample = True,
#                  crop_min = None,
#                  crop_max = None,
#                  only_state=False,
#                  num_demos=None, 
#                  random_augmentation=False,
#                  std_noise=0.0,
#                  validation_split=0.1,
#                  add_arm_pointcloud=True,
#                  arm_num_points=10000,
#                  cfg=None
#                  ):
#         self.only_state = only_state
#         self.num_demos = num_demos
#         self.folder_name = folder_name
#         self.voxel_size = voxel_size
#         self.downsample = downsample
#         self.crop_min = crop_min
#         self.crop_max = crop_max
#         self.add_arm_pointcloud = add_arm_pointcloud
#         self.arm_num_points = arm_num_points

#         self.num_points = num_points
#         self.pad_points = pad_points
#         self.std_noise = std_noise
#         self.random_augmentation = random_augmentation
#         self.from_disk = cfg['from_disk']
#         self.urdf = create_panda_urdf()
#         self.cfg=cfg
#         # load all actions
#         print("foler name", self.folder_name)
#         env_folder_name = folder_name.split("/")[-1]

#         if self.only_state:
#             if os.path.exists(f"demos/{env_folder_name}/demo_{0}_states.npy"):
#                 state = np.load(f"demos/{env_folder_name}/demo_{0}_states.npy")
#                 obs_dim = len(state[0])//3
#                 max_path_length = cfg['max_path_length']
#                 self.actions = np.zeros((num_demos, max_path_length))
#                 self.states = np.zeros((num_demos, max_path_length, obs_dim))
#                 self.traj_length = np.zeros(num_demos)
#                 for i in range(self.num_demos):
#                     states = np.load(f"demos/{env_folder_name}/demo_{i}_states.npy")
#                     traj_length = len(states)
#                     self.actions[i,:traj_length]=np.load(f"demos/{env_folder_name}/demo_{i}_actions.npy").astype(np.int)
#                     self.states[i, :traj_length]= states[:,:len(states[0])//3]
#                     self.traj_length[i]= traj_length
#                 self.num_trajs = num_demos
#             else:
#                 self.actions = np.load(f"{self.folder_name}/demo_actions.npy").astype(np.int)
#                 self.states = np.load(f"{self.folder_name}/demo_states.npy")
#                 self.joints = np.load(f"{self.folder_name}/demo_joints.npy")
#                 self.traj_length = np.array([self.states.shape[1] for i in range(self.states.shape[0])]).astype(np.int)
#                 self.num_trajs = self.states.shape[0]

#         else:
#             self.actions = np.load(f"{self.folder_name}/demo_actions.npy").astype(np.int)
#             self.states = np.load(f"{self.folder_name}/demo_states.npy")
#             self.joints = np.load(f"{self.folder_name}/demo_joints.npy")
            
#             self.traj_length = np.array([self.states.shape[1] for i in range(self.states.shape[0])]).astype(np.int)
#             self.num_trajs = self.states.shape[0]
#             if self.num_demos is not None:
#                 self.num_trajs = self.num_demos
            

#             # self.pcd_feats = np.zeros((self.num_demos, self.traj_length[0], self.num_points + self.arm_num_points, 1))
#             if not self.from_disk:
#                 if cfg['preload'] > 0:
#                     # import IPython
#                     # IPython.embed()
#                     self.pcd_points = np.load(f"{self.folder_name}/all_points.npy")
#                     # self.pcd_points = np.concatenate([self.pcd_points, np.zeros((self.num_demos - cfg['preload'], self.traj_length[0], self.num_points + self.arm_num_points, 3))])
#                     # pcd_points_preload = np.load(f"{self.folder_name}/all_points.npy")
#                     # self.pcd_points[:cfg['preload']] = pcd_points_preload[:cfg['preload']]
#                     # self.pcd_feats = np.load(f"{self.folder_name}/all_colors.npy")
#                 else:
#                     self.pcd_points = np.zeros((self.num_demos, self.traj_length[0], self.num_points + self.arm_num_points, 3))

#                 traj_idx = cfg['preload']
#                 for t in range(self.num_trajs-cfg['preload']):
#                     traj_idx = t + cfg['preload']
#                     start = time.time()
#                     traj_points = np.load(f"{self.folder_name}/traj_{traj_idx}_points.npy")

#                     for time_idx in range(len(self.states[traj_idx])):
#                         points = traj_points[time_idx]
#                         # import IPython
#                         # IPython.embed()
#                         # colors = np.load(f"{self.folder_name}/traj_{traj_idx}/{time_idx}_feats.npy")
#                         # points, colors = preprocess_points_feats(points, colors, self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample) #self.preprocess_pcd(pcd)
#                         # pcd = o3d.geometry.PointCloud()
#                         # pcd.points = o3d.utility.Vector3dVector(points)
#                         # o3d.visualization.draw_geometries([pcd])

#                         points, _ = crop_points_feats(points, np.ones(len(points)), crop_min, crop_max)
#                         points, _ = downsample_internal(points, np.ones(len(points)), num_points)
#                         points = add_arm_pcd(points, self.joints[traj_idx, time_idx], self.urdf, self.arm_num_points)

#                         self.pcd_points[traj_idx,time_idx] = points
#                         # self.pcd_feats[traj_idx,time_idx] = np.ones((len(points),1))
#                     print("Loaded traj", traj_idx, time.time() - start)
#                     wandb.log({"loaded_traj":traj_idx})
#                     if (traj_idx + 1) % cfg['store_freq'] == 0:
#                         np.save(f"{self.folder_name}/all_points.npy", self.pcd_points)
#                         np.save(f"{self.folder_name}/all_points_save.npy", self.pcd_points)
#                 if cfg['preload'] -  self.num_trajs != 0 and cfg['store_freq'] < 9000: 
#                     np.save(f"{self.folder_name}/all_points.npy", self.pcd_points)
#                     np.save(f"{self.folder_name}/all_points_save.npy", self.pcd_points)
#         self.traj_length = self.traj_length.astype(np.int)
#         self.actions = self.actions.astype(np.int)
#         self.obs_size = len(self.states[0][0])
#         split = np.random.random(self.num_trajs)< validation_split
#         self.validation_idx = np.where(split)
#         self.all_idx = np.where(np.logical_not(split))
    
#     def sample(self, batch_size, validation=False):
#         if validation:
#             traj_idxs = np.random.choice(self.validation_idx[0], batch_size)
#         else:
#             traj_idxs = np.random.choice(self.all_idx[0], batch_size)

#         traj_length = self.traj_length[traj_idxs]
#         time_idxs = np.array([np.random.choice(traj_length[i],1)[0] for i in range(batch_size)])
#         batch_actions = self.actions[traj_idxs, time_idxs]
#         if self.only_state:
#             batch_states = self.states[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
#         else:
#             batch_states = self.joints[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
#         batch_points = []
#         batch_feats = []
#         start = time.time()
#         if not self.only_state:
#             if not self.from_disk:
#                 batch_points = self.pcd_points[traj_idxs,time_idxs]
#             else:
#                 batch_points = []
#                 for traj_idx, time_idx in zip(traj_idxs, time_idxs):
#                     traj_points = np.load(f"{self.folder_name}/traj_{traj_idx}_points.npy")
#                     points = traj_points[time_idx]
#                     # import IPython
#                     # IPython.embed()
#                     # colors = np.load(f"{self.folder_name}/traj_{traj_idx}/{time_idx}_feats.npy")
#                     # points, colors = preprocess_points_feats(points, colors, self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample) #self.preprocess_pcd(pcd)
#                     # pcd = o3d.geometry.PointCloud()
#                     # pcd.points = o3d.utility.Vector3dVector(points)
#                     # o3d.visualization.draw_geometries([pcd])

#                     points, _ = crop_points_feats(points, np.ones(len(points)), self.crop_min, self.crop_max)
#                     points, _ = downsample_internal(points, np.ones(len(points)), self.num_points)
#                     points = add_arm_pcd(points, self.joints[traj_idx, time_idx], self.urdf, self.arm_num_points)
#                     batch_points.append(points) 
#                 batch_points = np.array(batch_points)

#             batch_feats = np.ones_like(batch_points)[...,0] #self.pcd_feats[traj_idxs,time_idxs]


#             if self.random_augmentation:
#                 batch_points = augment_pcds(batch_points, self.cfg)

#             if self.cfg["pcd_encoder_type"] != "pointnet":
#                 batch_points, batch_feats = preprocess_points_feats(batch_points, batch_feats, self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample)
#             else:
#                 batch_points, batch_feats = preprocess_points_feats_pointnet(batch_points, batch_feats, self.crop_min,self.crop_max,self.voxel_size, rgb_feats=False,num_points=self.num_points, add_padding=self.pad_points, downsample_points=self.downsample)
#         print(f"Buffer sampling", time.time() - start)

#         return batch_points, batch_feats, batch_states, batch_actions
from render_mesh_utils import PointCloudRenderer
from math import floor
class OnlineBuffer:
    def __init__(self, 
                 validation_split=0.1,
                 cfg=None,
                 ):
        
        self.urdf = create_panda_urdf(cfg)
        self.validation_split = validation_split
        self.num_trajs = 0
        self.cfg = cfg


    def sample_idxs(self,  batch_size, val_traj_idxs_batches=None):


        self.num_trajs, self.path_length = self.actions.shape

        buffer_size = self.num_trajs * self.path_length

        split = np.zeros(buffer_size)
        split[:] = False
        
        if val_traj_idxs_batches:

            split[np.concatenate(val_traj_idxs_batches)] = True
        else:
            if self.num_trajs == 1: # no fixed validation
                split = np.random.random(buffer_size) < self.validation_split
            else:
                split[:floor(buffer_size*self.validation_split)] = True

        # split = np.random.random(buffer_size) < self.validation_split
        # split[0] = True
        validation_idx = np.where(split)[0]
        train_idx = np.where(np.logical_not(split))[0]

        np.random.shuffle(train_idx)
        np.random.shuffle(validation_idx)

        train_idx_batches = []
        val_idx_batches = []
        i=0
        while i < len(train_idx):
            train_idx_batches.append(np.array(train_idx[i:i+batch_size]))
            i+= batch_size
        i=0
        while i < len(validation_idx):
            val_idx_batches.append(np.array(validation_idx[i:i+batch_size]))
            i+= batch_size


        return train_idx_batches, val_idx_batches
    
    @torch.no_grad()
    def sample(self, data_idx):

        traj_idxs = (data_idx/self.path_length).astype(np.int32)
        time_idxs = data_idx % self.path_length

        batch_actions = self.actions[traj_idxs, time_idxs]
        
        if self.expert_actions is not None:
            expert_actions = self.expert_actions[traj_idxs, time_idxs]
        else:
            expert_actions = None

        batch_joints = self.joints[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
        
        batch_full_states = self.states[traj_idxs, time_idxs]
        batch_states = extract_state(batch_full_states)#self.states[traj_idxs, time_idxs]
        if self.pcd_points is not None:
            batch_points = self.pcd_points[traj_idxs,time_idxs]

            if self.cfg["pcd_encoder_type"] == "pointnet":
                assert False
                batch_points, batch_feats = preprocess_points_feats_pointnet(batch_points, batch_feats, self.cfg["crop_min"],self.cfg["crop_max"],self.cfg["voxel_size"], rgb_feats=False,num_points=self.cfg["num_points"], add_padding=self.cfg["pad_points"], downsample_points=self.cfg["downsample"])

            else:
                # if self.cfg["use_synthetic_pcd"]:  
                #     batch_points, batch_feats = postprocess_pcd_synthetic(batch_points, self.cfg)
                # else:
                batch_points, batch_feats = postprocess_pcd(batch_points, self.cfg)
                # TODO: when generating pcd normally, I should add the arm already on the visualization, not on the processing
        else:
            batch_points = None
            batch_feats = None
           



        return batch_points, batch_feats, batch_states, batch_joints, batch_actions, expert_actions, batch_full_states
    
    def add_trajectories(self, actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors, expert_actions=None):
        if len(actions) == 0:
            return
            
        if self.num_trajs == 0:
            self.actions = actions
            self.cont_actions = cont_actions
            self.states = states
            self.joints = joints
            
            # if self.cfg["use_synthetic_pcd"]:
            #     all_pcd_points = []
            #     start = time.time()
            #     for i in range(states.shape[0]):
            #         all_pcd_points.append(self.synthetic_pcd.generate_pcd(states[i], torch.tensor(joints[i])))
            #     print("time to generate pcd", time.time() - start)
            #     self.pcd_points = np.array(all_pcd_points)
            # else:
            self.pcd_points = all_pcd_points

            self.traj_length = np.array([actions.shape[1] for _ in range(actions.shape[0])])
            self.expert_actions = expert_actions
        else:
            self.actions = np.concatenate([self.actions, actions])
            self.cont_actions = np.concatenate([self.cont_actions, cont_actions])
            self.states = np.concatenate([self.states, states])
            self.joints = np.concatenate([self.joints, joints])

            # if self.cfg["use_synthetic_pcd"]:

            #     all_pcd_points = []
            #     start = time.time()
            #     for i in range(states.shape[0]):
            #         all_pcd_points.append(self.synthetic_pcd.generate_pcd(states[i], torch.tensor(joints[i])))
            #     all_pcd_points = np.array(all_pcd_points)
            #     self.pcd_points = np.concatenate([self.pcd_points, np.array(all_pcd_points)])
            #     print("time to generate pcd", time.time() - start)


            if all_pcd_points is not None:
                self.pcd_points = np.concatenate([self.pcd_points, all_pcd_points])
                # self.pcd_colors = np.concatenate([self.pcd_colors, all_pcd_colors])


            self.traj_length = np.concatenate([self.traj_length, np.array([actions.shape[1] for _ in range(actions.shape[0])])])
            if self.expert_actions is not None:
                self.expert_actions = np.concatenate([self.expert_actions, expert_actions])

        self.num_trajs = len(states)
        split = np.random.random(self.num_trajs)< self.validation_split
        split[0] = True
        self.validation_idx = np.where(split)[0]
        self.train_idx = np.where(np.logical_not(split))[0]
    
    def reset(self):
        self.num_trajs = 0
    
    def store(self, num, foldername, node=0, main_folder="/scratch/marcel/data/online_distill"):
        os.makedirs(f"{main_folder}/{foldername}/", exist_ok=True)
        np.save(f"{main_folder}/{foldername}/actions_{node}_{str(num)}.npy", self.actions)
        np.save(f"{main_folder}/{foldername}/joints_{node}_{str(num)}.npy", self.joints)
        np.save(f"{main_folder}/{foldername}/states_{node}_{str(num)}.npy", self.states)
        # np.store(self.cont_actions, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        # np.store(self.states, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        if self.pcd_points is not None:
            np.save(f"{main_folder}/{foldername}/pcd_points_{node}_{str(num)}.npy", self.pcd_points)
        # np.save(f"{main_folder}/{foldername}/pcd_colors_{str(num)}.npy", self.pcd_colors)
        # np.save(f"{main_folder}/{foldername}/traj_length_{str(num)}.npy", self.traj_length)

class OnlineBufferPPO:
    def __init__(self, 
                 obs_dim,
                 validation_split=0.1,
                 cfg=None,
                 ):
        
        self.urdf = create_panda_urdf(cfg)
        self.validation_split = validation_split
        self.cfg = cfg
        self.num_demos = cfg["num_demos"]
        self.num_trajs = 0
        self.max_path_length = cfg["max_path_length"]
        self.actions = np.zeros((self.num_demos, self.max_path_length), dtype=np.int) 
        self.cont_actions = np.zeros((self.num_demos, self.max_path_length), dtype=np.int) 
        self.from_vision = cfg["from_vision"]
        if self.from_vision:
            obs_dim = 9
            self.joints = np.zeros((self.num_demos, self.max_path_length, obs_dim), dtype=np.float) 

        self.states = np.zeros((self.num_demos, self.max_path_length, obs_dim), dtype=np.float) 
        self.traj_lengths = np.zeros(self.num_demos, dtype=np.int)

        if cfg["from_vision"]:
            self.pcd_points =  np.zeros((self.num_demos, self.max_path_length, cfg["num_points_demos"], 3), dtype=np.float)
        else:
            self.pcd_points = None


    def sample_idxs(self,  batch_size, val_traj_idxs_batches=None):
        if val_traj_idxs_batches:
            split[np.concatenate(val_traj_idxs_batches)] = True
        else:
            self.indices = []
            for n in range(self.num_trajs):
                for t in range(self.traj_lengths[n]):
                    self.indices.append((n,t))
            self.indices = np.array(self.indices)
            buffer_size = len(self.indices)
            split = np.zeros(buffer_size)
            split[:] = False

            if self.num_trajs == 1: # no fixed validation
                split = np.random.random(buffer_size) < self.validation_split
            else:
                split[:floor(buffer_size*self.validation_split)] = True

        # split = np.random.random(buffer_size) < self.validation_split
        # split[0] = True
        validation_idx = np.where(split)[0]
        train_idx = np.where(np.logical_not(split))[0]

        np.random.shuffle(train_idx)
        np.random.shuffle(validation_idx)

        train_idx_batches = []
        val_idx_batches = []
        i=0
        while i < len(train_idx):
            train_idx_batches.append(np.array(train_idx[i:i+batch_size]))
            i+= batch_size
        i=0
        while i < len(validation_idx):
            val_idx_batches.append(np.array(validation_idx[i:i+batch_size]))
            i+= batch_size

        
        return train_idx_batches, val_idx_batches
    
    @torch.no_grad()
    def sample(self, data_idx):
        traj_idxs, time_idxs = self.indices[data_idx].transpose()

        batch_actions = self.actions[traj_idxs, time_idxs]
        
        # if self.expert_actions is not None:
        #     expert_actions = self.expert_actions[traj_idxs, time_idxs]
        # else:
        expert_actions = None

        if self.from_vision:
            batch_joints = self.joints[traj_idxs, time_idxs]#self.states[traj_idxs, time_idxs]
        else:
            batch_joints = None

        batch_full_states = self.states[traj_idxs, time_idxs]
        batch_states = extract_state(batch_full_states)#self.states[traj_idxs, time_idxs]
        if self.pcd_points is not None:
            batch_points = self.pcd_points[traj_idxs,time_idxs]

            if self.cfg["pcd_encoder_type"] == "pointnet":
                assert False
                batch_points, batch_feats = preprocess_points_feats_pointnet(batch_points, batch_feats, self.cfg["crop_min"],self.cfg["crop_max"],self.cfg["voxel_size"], rgb_feats=False,num_points=self.cfg["num_points"], add_padding=self.cfg["pad_points"], downsample_points=self.cfg["downsample"])

            else:
                # if self.cfg["use_synthetic_pcd"]:  
                #     batch_points, batch_feats = postprocess_pcd_synthetic(batch_points, self.cfg)
                # else:
                batch_points, batch_feats = postprocess_pcd(batch_points, self.cfg)
                # TODO: when generating pcd normally, I should add the arm already on the visualization, not on the processing
        else:
            batch_points = None
            batch_feats = None
           

        return batch_points, batch_feats, batch_states, batch_joints, batch_actions, expert_actions, batch_full_states
    
    def add_trajectories(self, actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors, expert_actions=None):
        if len(actions) == 0:
            return
            
        for idx, action in enumerate(actions):
            traj_length = states[idx].shape[0]
            self.actions[self.num_trajs, :traj_length] = actions[idx]
            self.cont_actions[self.num_trajs, :traj_length] = cont_actions[idx]
            self.states[self.num_trajs, :traj_length] = states[idx]
            if self.from_vision:
                self.joints[self.num_trajs, :traj_length] = joints[idx]

            # if self.cfg["use_synthetic_pcd"]:

            #     all_pcd_points = []
            #     start = time.time()
            #     for i in range(states.shape[0]):
            #         all_pcd_points.append(self.synthetic_pcd.generate_pcd(states[i], torch.tensor(joints[i])))
            #     all_pcd_points = np.array(all_pcd_points)
            #     self.pcd_points = np.concatenate([self.pcd_points, np.array(all_pcd_points)])
            #     print("time to generate pcd", time.time() - start)


            if all_pcd_points is not None:
                    
                downsampled_points = []
                for i in range(all_pcd_points[idx].shape[0]):
                    indices = np.random.choice(all_pcd_points[idx][0].shape[0], int(self.cfg["num_points_demos"]), replace=False)
                    
                    downsampled_points.append(all_pcd_points[idx, i, indices])
                
                self.pcd_points[self.num_trajs, :traj_length] =downsampled_points
                # self.pcd_colors = np.concatenate([self.pcd_colors, all_pcd_colors])


            self.traj_lengths[self.num_trajs] = traj_length

            if expert_actions is not None:
                self.expert_actions[self.num_trajs, :traj_length] = expert_actions[idx]
            
            self.num_trajs += 1
    
    def reset(self):
        self.num_trajs = 0
    
    def store(self, num, foldername, node=0, main_folder="/scratch/marcel/data/online_distill"):
        os.makedirs(f"{main_folder}/{foldername}/", exist_ok=True)
        np.save(f"{main_folder}/{foldername}/actions_{node}_{str(num)}.npy", self.actions)
        np.save(f"{main_folder}/{foldername}/joints_{node}_{str(num)}.npy", self.joints)
        np.save(f"{main_folder}/{foldername}/states_{node}_{str(num)}.npy", self.states)
        # np.store(self.cont_actions, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        # np.store(self.states, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        if self.pcd_points is not None:
            np.save(f"{main_folder}/{foldername}/pcd_points_{node}_{str(num)}.npy", self.pcd_points)
        # np.save(f"{main_folder}/{foldername}/pcd_colors_{str(num)}.npy", self.pcd_colors)
        # np.save(f"{main_folder}/{foldername}/traj_length_{str(num)}.npy", self.traj_length)


class OnlineBufferHistory:
    def __init__(self, 
                 validation_split=0.1,
                 cfg=None,
                 ):
        
        self.urdf = create_panda_urdf(cfg)
        self.validation_split = validation_split
        self.num_trajs = 0
        self.cfg = cfg

    def sample_idxs(self,  batch_size, val_traj_idxs_batches=None):
        self.num_trajs, self.path_length = self.actions.shape

        buffer_size = self.num_trajs

        if val_traj_idxs_batches:
            split = np.zeros(buffer_size)
            split[:] = False
            split[np.concatenate(val_traj_idxs_batches)] = True
        else:
            split = np.random.random(buffer_size) < self.validation_split
            split[0] = True

        # split = np.random.random(buffer_size) < self.validation_split
        # split[0] = True
        validation_idx = np.where(split)[0]
        train_idx = np.where(np.logical_not(split))[0]

        np.random.shuffle(train_idx)
        np.random.shuffle(validation_idx)

        train_idx_batches = []
        val_idx_batches = []
        i=0
        while i < len(train_idx):
            train_idx_batches.append(np.array(train_idx[i:i+batch_size]))
            i+= batch_size
        i=0
        while i < len(validation_idx):
            val_idx_batches.append(np.array(validation_idx[i:i+batch_size]))
            i+= batch_size

        return train_idx_batches, val_idx_batches

    def sample(self, data_idx):
        traj_idxs = (data_idx).astype(np.int32)

        batch_actions = self.actions[traj_idxs, :]
        
        if self.expert_actions is not None:
            expert_actions = self.expert_actions[traj_idxs, :]
        else:
            expert_actions = None

        batch_joints = self.joints[traj_idxs, :]#self.states[traj_idxs, time_idxs]

        batch_states = self.states[traj_idxs, :]#self.states[traj_idxs, time_idxs]

        
        if self.pcd_points is not None:
            batch_points = self.pcd_points[traj_idxs,:]
            
            batch_feats = np.ones_like(batch_points)[...,0].reshape((*batch_points.shape[:-1], 1)) #self.pcd_feats[traj_idxs,time_idxs]
            # if self.random_augmentation:
            batch_points = augment_pcds_with_history(batch_points, self.cfg)

            batch_points, batch_feats = preprocess_points_feats_history(batch_points, batch_feats, self.cfg["crop_min"],self.cfg["crop_max"],self.cfg["voxel_size"], rgb_feats=False,num_points=self.cfg["num_points"], add_padding=self.cfg["pad_points"], downsample_points=self.cfg["downsample"])

        else:
            batch_points = None
            batch_feats = None
        return batch_points, batch_feats, batch_states, batch_joints, batch_actions, expert_actions
    
    def add_trajectories(self, actions, cont_actions, states, joints, all_pcd_points, all_pcd_colors, expert_actions=None):
        if self.num_trajs == 0:
            self.actions = actions
            self.cont_actions = cont_actions
            self.states = states
            self.joints = joints
            self.pcd_points = all_pcd_points
            self.pcd_colors = all_pcd_colors
            self.traj_length = np.array([actions.shape[1] for _ in range(actions.shape[0])])
            self.expert_actions = expert_actions
        else:
            self.actions = np.concatenate([self.actions, actions])
            self.cont_actions = np.concatenate([self.cont_actions, cont_actions])
            self.states = np.concatenate([self.states, states])
            self.joints = np.concatenate([self.joints, joints])
            if all_pcd_points is not None:
                self.pcd_points = np.concatenate([self.pcd_points, all_pcd_points])
                self.pcd_colors = np.concatenate([self.pcd_colors, all_pcd_colors])
            self.traj_length = np.concatenate([self.traj_length, np.array([actions.shape[1] for _ in range(actions.shape[0])])])
            if self.expert_actions is not None:
                self.expert_actions = np.concatenate([self.expert_actions, expert_actions])

        self.num_trajs = len(states)
        split = np.random.random(self.num_trajs)< self.validation_split
        split[0] = True
        self.validation_idx = np.where(split)[0]
        self.train_idx = np.where(np.logical_not(split))[0]
    
    def reset(self):
        self.num_trajs = 0
    
    def store(self, num, foldername, main_folder="/scratch/marcel/data/online_distill"):
        os.makedirs(f"{main_folder}/{foldername}/", exist_ok=True)
        np.save(f"{main_folder}/{foldername}/actions_{str(num)}.npy", self.actions)
        np.save(f"{main_folder}/{foldername}/joints_{str(num)}.npy", self.joints)
        np.save(f"{main_folder}/{foldername}/states_{str(num)}.npy", self.states)
        # np.store(self.cont_actions, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        # np.store(self.states, f"{main_folder}/{foldername}/actions_{str(num)}.npy")
        np.save(f"{main_folder}/{foldername}/pcd_points_{str(num)}.npy", self.pcd_points)
        # np.save(f"{main_folder}/{foldername}/pcd_colors_{str(num)}.npy", self.pcd_colors)
        # np.save(f"{main_folder}/{foldername}/traj_length_{str(num)}.npy", self.traj_length)