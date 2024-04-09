import torch
from utils import add_arm_pcd, create_panda_urdf
import numpy as np
import time

class PointCloudRenderer:
    def __init__(self, env, cfg=None):
        
        self.all_mesh_pcd = []
        self.mesh_names = cfg["mesh_names"]

        self.total_objects = cfg["total_objects"]
        self.object_idxs = cfg["object_idxs"]
        self.coms_offsets = torch.tensor([[-0.0005,  0.1614,  0.0263], [0.0406, 0.0924, 0.0435],[-0.0402,  0.2218,  0.0063]]).to("cuda").float() # TODO this should be removed too
        for mesh_name in self.mesh_names:
            self.all_mesh_pcd.append(np.load(f"{folder}/{mesh_name}_pcd.npy"))

        self.urdf = create_panda_urdf()

    def generate_pcd(self, state, joints=None):
        start = time.time()
        from omni.isaac.orbit.utils.math import quat_inv, quat_apply, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
        
        all_pcds = []
        num_points = 1000
        pcd = []
        state = torch.tensor(state).to("cuda").float()

        for idx, mesh_name in enumerate(self.mesh_names):
            
            # print("coms", env._wrapped_env.base_env._env.scene.object_coms[mesh_name][0])
            obj_idx = self.object_idxs[idx]
            pos = state[:, obj_idx*3:(obj_idx+1)*3] - self.coms_offsets[idx]
            rot = state[:, self.total_objects*3 + obj_idx*4:self.total_objects*3 +(obj_idx+1)*4]
            choice = np.random.choice(len(self.all_mesh_pcd[idx]), num_points , replace=False)
            mesh = self.all_mesh_pcd[idx][choice]
            # mesh = [ mesh for i in range(100)]
            final_shape = (state.shape[0], *mesh.shape)
            mesh = np.broadcast_to(mesh, final_shape)

            rot = np.broadcast_to(rot.cpu().numpy(), (num_points,state.shape[0],4)).transpose(1,0,2)
            rotated_points = quat_apply( torch.tensor(rot).to("cuda").float(), torch.tensor(mesh).to("cuda").float())
            trans_points = rotated_points + pos.unsqueeze(1)
            pcd.append(trans_points.cpu().detach().numpy())
        pcd = np.concatenate(pcd, axis=1)
        all_pcds = []
        for i in range(state.shape[0]):

            all_pcds.append(add_arm_pcd(pcd[i], joints[i].detach().cpu().numpy(), self.urdf, num_points*3))

        print("Time to generate pcdds", state.shape[0], time.time() - start)
        return np.array(all_pcds)

class PointCloudRendererOnline:
    def __init__(self, env, cfg=None):
        from omni.isaac.core.prims import RigidPrimView
        self.env = env
        self.all_mesh_pcd = []
        self.all_arm_mesh_pcd = []
        self.mesh_names = cfg["mesh_names"]
        self.arm_names = ['panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 
    'panda_link6', 'panda_link7', 'panda_hand', 'panda_leftfinger', 'panda_rightfinger']
        # self.total_objects = cfg["total_objects"]
        # self.object_idxs = cfg["object_idxs"]
        # self.coms_offsets = torch.tensor([[-0.0005,  0.1614,  0.0263], [0.0406, 0.0924, 0.0435],[-0.0402,  0.2218,  0.0063]]).to("cuda").float() # TODO this should be removed too
        for mesh_name in self.mesh_names:
            self.all_mesh_pcd.append(np.load(f"{cfg['foldermeshname']}/{mesh_name}_pcd.npy"))
        self.arm_views = []
        for mesh_name in self.arm_names:
            self.all_arm_mesh_pcd.append(np.load(f"franka_arm_meshes/{mesh_name}_pcd.npy"))

            view = RigidPrimView(
                        prim_paths_expr=f"/World/envs/env_*/Robot/{mesh_name}", reset_xform_properties=False
            )
            view.initialize()
            self.arm_views.append(view)
    

    def generate_pcd(self,):
        start = time.time()

        from omni.isaac.orbit.utils.math import quat_inv, quat_apply, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
        
        all_pcds = []
        num_points = 1000
        pcd = []

        data = self.env._env.scene._data
        num_envs = self.env.num_envs
        for idx, mesh_name in enumerate(self.mesh_names):
            
            # print("coms", env._wrapped_env.base_env._env.scene.object_coms[mesh_name][0])
            # obj_idx = self.object_idxs[idx]

            # TODO: get poses of the object from the environment
            pose = data[mesh_name]
            # pos = state[:, obj_idx*3:(obj_idx+1)*3] - self.coms_offsets[idx] 
            # rot = state[:, self.total_objects*3 + obj_idx*4:self.total_objects*3 +(obj_idx+1)*4]
            pos = pose.root_pos_w - self.env._env.envs_positions
            rot = pose.root_quat_w
            choice = np.random.choice(len(self.all_mesh_pcd[idx]), num_points , replace=False)
            mesh = self.all_mesh_pcd[idx][choice]
            # mesh = [ mesh for i in range(100)]
            final_shape = (num_envs, *mesh.shape)
            mesh = np.broadcast_to(mesh, final_shape)

            rot = np.broadcast_to(rot.cpu().numpy(), (num_points,num_envs,4)).transpose(1,0,2)
            rotated_points = quat_apply( torch.tensor(rot).to("cuda").float(), torch.tensor(mesh).to("cuda").float())
            trans_points = rotated_points + pos.unsqueeze(1)
            pcd.append(trans_points.cpu().detach().numpy())
        
        arm_num_points = 300
        for idx, mesh_name in enumerate(self.arm_names):
            # print("coms", env._wrapped_env.base_env._env.scene.object_coms[mesh_name][0])
            pose = self.arm_views[idx].get_world_poses() 
            pos = pose[0]- self.env._env.envs_positions
            rot = pose[1]

            choice = np.random.choice(len(self.all_arm_mesh_pcd[idx]), arm_num_points , replace=False)
            mesh = self.all_arm_mesh_pcd[idx][choice]
            # mesh = [ mesh for i in range(100)]
            final_shape = (num_envs, *mesh.shape)
            mesh = np.broadcast_to(mesh, final_shape)
            
            rot = np.broadcast_to(rot.cpu().numpy(), (arm_num_points,num_envs,4)).transpose(1,0,2)
            rotated_points = quat_apply( torch.tensor(rot).to("cuda").float(), torch.tensor(mesh).to("cuda").float())
            trans_points = rotated_points + pos.unsqueeze(1)
            pcd.append(trans_points.cpu().detach().numpy())
        
        pcd = np.concatenate(pcd, axis=1)

        print("Time to generate pcdds", num_envs, time.time() - start)
        return pcd
        # import open3d as o3d
        # all_points = pcd
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_points))
        # o3d.visualization.draw_geometries([pcd])