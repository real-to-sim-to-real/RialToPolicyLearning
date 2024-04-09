# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gym.spaces
import math
import torch
from typing import List
import copy

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.semantics as semantics_utils
import time
import omni.isaac.orbit.utils.kit as kit_utils
# from omni.isaac.orbit.controllers.differential_inverse_kinematics import DifferentialInverseKinematics
from omni.isaac.orbit.markers import StaticMarker
# from omni.isaac.orbit.objects import RigidObject
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.dict import class_to_dict
from omni.isaac.orbit.utils.math import quat_inv, quat_mul,euler_xyz_from_quat, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
from omni.isaac.orbit.utils.mdp import ObservationManager, RewardManager
import numpy as np
# from omni.isaac.orbit.objects.articulated import ArticulatedObject

from omni.isaac.orbit_envs.isaac_env import VecEnvIndices, VecEnvObs
from rialto.envs.general.isaac_env_general import IsaacEnvGeneral
# from omni.isaac.orbit.controllers.differential_inverse_kinematics import (
#     DifferentialInverseKinematics,
#     DifferentialInverseKinematicsCfg,
# )
from rialto.envs.general.diff_ik import DifferentialInverseKinematics, DifferentialInverseKinematicsCfg
from .general_cfg import GeneralEnvCfg
from rialto.envs.utils.scene import SceneObject
from omni.isaac.orbit.utils.math import (
    quat_apply,
    quat_mul,
    matrix_from_quat
)
import wandb
from rialto.envs.general.franka_config import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
# from huge.envs.general.custom_pb_ik_solver import CustomPBIKSolver
# from huge.envs.general.qp_ik_solver import QPDifferentialInverseKinematics

QP_SOLVER = 0
PB_SOLVER = 1
ISAAC_PINV_SOLVER = 2
class GeneralEnv(IsaacEnvGeneral):
    """Environment for lifting an object off a table with a single-arm manipulator."""

    def __init__(self, cfg: GeneralEnvCfg = None, headless: bool = False):
        # copy configuration
        self.cfg = cfg
        # parse the configuration for controller configuration
        # note: controller decides the robot control mode
        self._pre_process_cfg()
        # create classes (these are called by the function :meth:`_design_scene`)
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
        # robot_cfg.init_state.pos = (0,0,-0.03)
        robot_cfg.data_info.enable_jacobian = True
        robot_cfg.rigid_props.disable_gravity = True
        # robot_cfg.init_state.dof_pos = {
        #     "panda_joint1": 9.8643256e-03,
        #     "panda_joint2": -5.4602885e-01,
        #     "panda_joint3": -1.0333774e-02,
        #     "panda_joint4": -2.7952669e+00,
        #     "panda_joint5": 1.9831273e-03,
        #     "panda_joint6": 3.0733757e+00,
        #     "panda_joint7": 7.2458827e-01,
        #     "panda_finger_joint*": 0.04,
        # }
      
        
        # spawn robot
        self.robot = SingleArmManipulator(cfg=robot_cfg)
        self.scene = SceneObject(self.cfg.scene)
        self.ik_solver_type = ISAAC_PINV_SOLVER

        # initialize the base class to setup the scene.
        super().__init__(self.cfg, headless=headless)

        # parse the configuration for information
        self._process_cfg()
        # initialize views for the cloned scenes
        self._initialize_views()
        self.usd_path = self.cfg.scene.meta_info.usd_path
        # prepare the observation manager
        self._observation_manager = SceneObservationManager(class_to_dict(self.cfg.observations), self, self.device)
        # prepare the reward manager
        self._reward_manager = SceneRewardManager(
            class_to_dict(self.cfg.rewards), self, self.num_envs, self.dt, self.device
        )
        # print information about MDP
        print("[INFO] Observation Manager:", self._observation_manager)
        print("[INFO] Reward Manager: ", self._reward_manager)

        # compute the observation space: arm joint state + ee-position + goal-position + actions
        num_obs = self._observation_manager.group_obs_dim["policy"][0]
        self.observation_space = gym.spaces.Box(low=-math.inf, high=math.inf, shape=(num_obs,))
        # compute the action space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        #self.all_actions = torch.zeros((self.num_envs, 250, self.num_actions)).to(self.device)
        print("[INFO]: Completed setting up the environment...")

        # Take an initial step to initialize the scene.
        # This is required to compute quantities like Jacobians used in step().
        self.sim.step()
        # -- fill up buffers
        self.scene.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)


    """
    Implementation specifics.
    """

    def _design_scene(self) -> List[str]:
        # ground plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane", size=1000, z_position=self.cfg.randomization.floor_height)

        
        # table
        # prim_utils.create_prim(self.template_env_ns + "/Table", usd_path=self.cfg.table.usd_path)
        # robot
        self.robot.spawn(self.template_env_ns + "/Robot", translation=(0.0, 0.0, 0.0))
        """
            max_velocity = 50
        for dof in self.cfg.robot.init_state.dof_pos:
            print(self.template_env_ns + "/Robot", dof)

            if "finger" in dof:
                dof = "panda_finger_joint1"
                kit_utils.set_drive_dof_properties(self.template_env_ns + "/Robot", dof, max_velocity=max_velocity)
                dof = "panda_finger_joint2"
                kit_utils.set_drive_dof_properties(self.template_env_ns + "/Robot", dof, max_velocity=max_velocity)

            else:
                kit_utils.set_drive_dof_properties(self.template_env_ns + "/Robot", dof, max_velocity=max_velocity)
        """

        # object
        self.scene.spawn(self.template_env_ns + "/Scene")

        semantics_utils.add_update_semantics(prim_utils.get_prim_at_path(self.template_env_ns + "/Robot"), "Robot")       
        semantics_utils.add_update_semantics(prim_utils.get_prim_at_path(self.template_env_ns + "/Scene"), "Scene")       

        # self.plate_marker = StaticMarker(
        #    "/Visuals/plate",
        #    self.num_envs,
        #    usd_path=self.cfg.frame_marker.usd_path,
        #    scale=self.cfg.frame_marker.scale,
        #)

        #self.dishrack_marker = StaticMarker(
        #    "/Visuals/dishrack",
        #    self.num_envs,
        #    usd_path=self.cfg.frame_marker.usd_path,
        #    scale=self.cfg.frame_marker.scale,
        #)

        # setup debug visualization
        print("Enable render", self.enable_render, self.cfg.viewer.debug_vis)
        # if self.cfg.viewer.debug_vis and self.enable_render:
            # create point instancer to visualize the goal points
            # self._goal_markers = StaticMarker(
            #     "/Visuals/object_goal",
            #     self.num_envs,
            #     usd_path=self.cfg.goal_marker.usd_path,
            #     scale=self.cfg.goal_marker.scale,
            # )
            # create marker for viewing end-effector pose
        # self._ee_markers = StaticMarker(
        #     "/Visuals/ee_current",
        #     self.num_envs,
        #     usd_path=self.cfg.frame_marker.usd_path,
        #     scale=self.cfg.frame_marker.scale,
        # )
        # # # create marker for viewing command (if task-space controller is used)
        # if self.cfg.control.control_type == "inverse_kinematics":
        #     self._cmd_markers = StaticMarker(
        #         "/Visuals/ik_command",
        #         self.num_envs,
        #         usd_path=self.cfg.frame_marker.usd_path,
        #         scale=self.cfg.frame_marker.scale,
        #     )
        # return list of global prims
        return ["/World/defaultGroundPlane"]
    
    def _debug_vis(self):
        """Visualize the environment in debug mode."""
        # apply to instance manager
        # -- goal
        # self._goal_markers.set_world_poses(self.object_des_pose_w[:, 0:3], self.object_des_pose_w[:, 3:7])
        # -- end-effector
        # self._ee_markers.set_world_poses(self.robot.data.ee_state_w[:, 0:3].clone(), self.robot.data.ee_state_w[:, 3:7].clone())
        # # -- task-space commands
        # if self.cfg.control.control_type == "inverse_kinematics":
        #     # convert to world frame
        #     # ee_positions = self._ik_controller.desired_ee_pos + self.envs_positions
        #     # ee_orientations = self._ik_controller.desired_ee_rot            
        #     ee_positions = self.actions[:, :3] + self.envs_positions
        #     ee_orientations = self.actions[:, 3:7]
            
        #     # set poses
        #     self._cmd_markers.set_world_poses(ee_positions, ee_orientations)


    def randomize_object_pose(self,env_ids):
        root_state = self.scene.get_default_root_state(env_ids)

        for idx, object_list in enumerate(self.cfg.randomization.randomize_object_name):
            first_rand = 0
            for idx_rand, object_name in enumerate(object_list):
                if idx_rand < len(self.cfg.randomization.object_initial_pose.position_min_bound[idx]):
                    lower = torch.tensor(self.cfg.randomization.object_initial_pose.position_min_bound[idx][idx_rand]).to(self.device)
                    upper = torch.tensor(self.cfg.randomization.object_initial_pose.position_max_bound[idx][idx_rand]).to(self.device)
                    # object_name = self.cfg.randomization.randomize_object_name

                    rand = sample_uniform(
                            lower, upper, (len(env_ids), 3), device=self.device
                        )
                else:
                    rand = 0

                root_state[object_name][:,:3] += first_rand + rand
                if idx_rand == 0:
                    first_rand = rand
                if self.cfg.randomization.randomize_rot:
                    print("Randomize rot")
                    if idx_rand < len(self.cfg.randomization.object_initial_pose.orientation_min_bound[idx]):
                        lower = torch.tensor(self.cfg.randomization.object_initial_pose.orientation_min_bound[idx][idx_rand]).to(self.device)
                        upper = torch.tensor(self.cfg.randomization.object_initial_pose.orientation_max_bound[idx][idx_rand]).to(self.device)
                    else:
                        lower = 0
                        upper = 0

                    rand_quat = self.random_yaw_orientation_with_bounds_object(lower, upper, len(env_ids))
                    
                    root_state[object_name][:,3:7] = quat_mul(rand_quat, root_state[object_name][:,3:7])

            # self.scene.objects["Object_268"].set_local_scales(np.array([[0.5,0.5,0.5]]))
            
        for distractor_name in self.scene.distractor_names:
                
            lower = torch.tensor(self.cfg.randomization.distractor_initial_pose.position_min_bound).to(self.device)
            upper = torch.tensor(self.cfg.randomization.distractor_initial_pose.position_max_bound).to(self.device)
            # object_name = self.cfg.randomization.randomize_object_name

            rand = sample_uniform(
                    lower, upper, (len(env_ids), 3), device=self.device
                )
            
            root_state[distractor_name][:,:3] += rand

            if self.cfg.randomization.randomize_rot:
                print("Randomize distractor rot")
                rand_quat = self.random_yaw_orientation_with_bounds_distractor(len(env_ids))

                root_state[distractor_name][:,3:7] = quat_mul(rand_quat, root_state[distractor_name][:,3:7])

        for object_name in root_state:
            if "joint" in object_name.lower():
                continue
            root_state[object_name][:,:3] += self.envs_positions[env_ids]

        self.scene.set_root_state(root_state, env_ids=env_ids)

    def randomize_robot_pose(self,env_ids):
        root_state = self.robot.get_default_root_state(env_ids=env_ids)
        lower = torch.tensor(self.cfg.randomization.robot_initial_pose.position_min_bound).to(self.device)
        upper = torch.tensor(self.cfg.randomization.robot_initial_pose.position_max_bound).to(self.device)
        root_state[:,:3] += sample_uniform(
                lower, upper, (len(env_ids), 3), device=self.device
            )
        root_state[:,:3] += self.envs_positions[env_ids]
        if self.cfg.randomization.randomize_rot:
            print("Randomize rot")
            root_state[:,3:7] = quat_mul(self.random_yaw_orientation_with_bounds(len(env_ids)), root_state[:,3:7])
        
        self.robot.set_root_state(root_state, env_ids=env_ids)
    
    def get_robot_frame(self):
        return self.robot._data.root_state_w
    
    def get_robot_joints(self):
        obs = self.robot.data.arm_dof_pos

        tool_obs = scale_transform(
            self.robot.data.tool_dof_pos,
            self.robot.data.soft_dof_pos_limits[:, self.robot.arm_num_dof :, 0],
            self.robot.data.soft_dof_pos_limits[:, self.robot.arm_num_dof :, 1],
        )

        return torch.hstack([obs, tool_obs])
    
    def random_yaw_orientation_with_bounds(self, num: int) -> torch.Tensor:
        """Returns sampled rotation around z-axis.

        Args:
            num (int): The number of rotations to sample.
            device (str): Device to create tensor on.

        Returns:
            torch.Tensor: Sampled quaternion (w, x, y, z).
        """
        lower = torch.tensor(self.cfg.randomization.robot_initial_pose.orientation_min_bound).to(self.device)
        upper = torch.tensor(self.cfg.randomization.robot_initial_pose.orientation_max_bound).to(self.device)
        roll = torch.zeros(num, dtype=torch.float, device=self.device)
        pitch = torch.zeros(num, dtype=torch.float, device=self.device)
        yaw = (upper - lower) * torch.rand(num, dtype=torch.float, device=self.device) + lower

        return quat_from_euler_xyz(roll, pitch, yaw)

    def random_yaw_orientation_with_bounds_object(self, lower, upper,  num: int) -> torch.Tensor:
        """Returns sampled rotation around z-axis.

        Args:
            num (int): The number of rotations to sample.
            device (str): Device to create tensor on.

        Returns:
            torch.Tensor: Sampled quaternion (w, x, y, z).
        """
        
        roll = torch.zeros(num, dtype=torch.float, device=self.device)
        pitch = torch.zeros(num, dtype=torch.float, device=self.device)
        yaw = (upper - lower) * torch.rand(num, dtype=torch.float, device=self.device) + lower

        return quat_from_euler_xyz(roll, pitch, yaw)
    
    def random_yaw_orientation_with_bounds_distractor(self,  num: int) -> torch.Tensor:
        """Returns sampled rotation around z-axis.

        Args:
            num (int): The number of rotations to sample.
            device (str): Device to create tensor on.

        Returns:
            torch.Tensor: Sampled quaternion (w, x, y, z).
        """
        lower = torch.tensor(self.cfg.randomization.distractor_initial_pose.orientation_min_bound).to(self.device)
        upper = torch.tensor(self.cfg.randomization.distractor_initial_pose.orientation_max_bound).to(self.device)
        roll = torch.zeros(num, dtype=torch.float, device=self.device)
        pitch = torch.zeros(num, dtype=torch.float, device=self.device)
        yaw = (upper - lower) * torch.rand(num, dtype=torch.float, device=self.device) + lower

        return quat_from_euler_xyz(roll, pitch, yaw)
    def _reset_idx(self, env_ids):
        print("reset", env_ids)
        # randomize the MDP
        # -- robot DOF state
        dof_pos, dof_vel = self.robot.get_default_dof_state(env_ids=env_ids)
        self.robot.set_dof_state(dof_pos, dof_vel, env_ids=env_ids)
        self.robot.reset_buffers(env_ids=env_ids)
        # -- object pose

        # if self.cfg.randomization.object_initial_pose.position_cat == "uniform":
        #     self._randomize_scene_initial_pose(env_ids=env_ids)
        # else:
        self._reset_scene_initial_pose(env_ids=env_ids)
        
        if self.cfg.randomization.randomize_pos:
            print("Randomize pos", env_ids)
            self.randomize_robot_pose(env_ids)
            self.randomize_object_pose(env_ids)

        # -- Reward logging
        # fill extras with episode information
        self.extras["episode"] = dict()
        # reset
        # -- rewards manager: fills the sums for terminated episodes
        self._reward_manager.reset_idx(env_ids, self.extras["episode"])
        # -- obs manager
        self._observation_manager.reset_idx(env_ids)
        # -- reset history
        self.previous_actions[env_ids] = 0
        # -- MDP reset
        self.reset_buf[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        #self.all_actions[env_ids, :,:] = 0 
        # controller reset
        if self.cfg.control.control_type == "inverse_kinematics":
            # self._ik_controller.reset_idx(env_ids)
            print("inverse ik")
        else:
            print("joint pos control")

        self.sim.step(render=self.enable_render)

        self.scene.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)

        idx_success = np.where(self.get_reward().detach().cpu())[0]
        if len(idx_success) != 0:
            self._reset_idx(torch.tensor(idx_success))

        # idx = self.check_collisions(env_ids)

        # print("idx", idx)


        # if len(idx) != 0:
        #     self._reset_idx(torch.tensor(idx))

    def check_collisions(self, env_ids=None):
        import omni.physx as omni_physx
        import omni.isaac.core.utils.prims as prim_utils
        from pxr import PhysicsSchemaTools 
        collided_idxs = []
        id_list = [268, 266]
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        for env_id in env_ids:
            for id in id_list:
                physx_query_interface = omni_physx.get_physx_scene_query_interface()
                book_prim = prim_utils.get_prim_at_path(f'/World/envs/env_{env_id}/Scene/Xform_{id}/Object_{id}/Geometry/Object_Geometry')
                # book_mesh_prim = book_prim.GetChildren()[0].GetChildren()[0]
                sdf_path_encoded = PhysicsSchemaTools.encodeSdfPath(book_prim.GetPath())
                # sdf_path_encoded = PhysicsSchemaTools.encodeSdfPath(book_prim.GetPath())
                
                self.collided = False
                # mesh_hit = physx_query_interface.overlap_mesh_any(sdf_path_encoded[0], sdf_path_encoded[1])
                mesh_hit = physx_query_interface.overlap_mesh(
                    sdf_path_encoded[0], 
                    sdf_path_encoded[1],
                    self.report_hit,
                    True)        
                if self.collided:
                    collided_idxs.append(env_id)
                    break

        return collided_idxs

    def report_hit(self, hit):
        # import omni.usd
        # from pxr import UsdGeom, Gf, Vt
        # stage = omni.usd.get_context().get_stage()
        # hitColor = Vt.Vec3fArray([Gf.Vec3f(180.0 / 255.0, 16.0 / 255.0, 0.0)])
        # usdGeom = UsdGeom.Mesh.Get(stage, hit.rigid_body)
        # usdGeom.GetDisplayColorAttr().Set(hitColor)
        print(f'hit: ', dir(hit))
        print(f'hit.collision: ', hit.collision)
        print(f'hit.protoIndex: ', hit.protoIndex)
        print(f'hit.rigid_body: ', hit.rigid_body)
        # print(f'new usdGeom ', usdGeom)
        if "Robot" in hit.rigid_body:
            self.collided = True
        return True
        # return False
    #def _move_to_gripper_frame(self, actions:torch.Tensor, gripper_rot:torch.Tensor):


    def convert_tip2wrist(self, robot_tip_pos, robot_tip_quat):
        pos_wrist_tip = torch.tensor(np.tile([0,0,-0.1034],(self.num_envs,1))).float().to(self.device)
        rot_wrist_tip = torch.tensor(np.tile([1, 0,0,0],(self.num_envs,1))).float().to(self.device)
        
        wrist_in_world = quat_apply(robot_tip_quat, pos_wrist_tip) + robot_tip_pos 

        return wrist_in_world, robot_tip_quat

    def convert_wrist2tip(self, robot_wrist_pos, robot_wrist_quat):
        pos_tip_wrist = torch.tensor(np.tile([0,0,0.1034],(self.num_envs,1))).float().to(self.device)
        rot_tip_wrist = torch.tensor(np.tile([1, 0,0,0],(self.num_envs,1))).float().to(self.device)
        
        tip_in_world = quat_apply(robot_wrist_quat, pos_tip_wrist) + robot_wrist_pos 

        return tip_in_world, robot_wrist_quat

    def _step_impl(self, actions: torch.Tensor, control_type=None):
        # idx = self.check_collisions()

        # print("collision", idx)
        if not control_type:
            control_type = self.cfg.control.control_type
        #self.all_actions[:, self.episode_length_buf, :] = actions
        robot_data_beg = self.robot.data.arm_dof_pos.clone()
        robot_data_ee_state = self.robot.data.ee_state_w.clone()

        # pre-step: set actions into buffer
        #actions = torch.clamp(actions.clone(), -1, 1)
        self.actions = actions.clone().to(device=self.device)
        self.previous_state = self._get_observations()
        # self.actions[:,-1] = self.previous_state['policy']['tool_dof_pos_scaled'][:,0]*(0.2-torch.abs(self.actions[:,-1])) + self.actions[:,-1]*5
        # transform actions based on controller
        if  control_type == "inverse_kinematics":
            robot_pos = self.robot.data.ee_state_w.clone()[:,:3] - self.envs_positions
            robot_rot = self.robot.data.ee_state_w.clone()[:,3:7]

            # obtain 4x4 transformation matrix representing wrist poses
            robot_wrist_pos, robot_wrist_quat = self.convert_tip2wrist(robot_pos, robot_rot)

            # apply delta actions in wrist space
            actions_pos, actions_rot, actions_finger = self.actions[:,:3],self.actions[:,3:7], self.actions[:,7].reshape(-1,1)
            wrist_pos_des = robot_wrist_pos + actions_pos
            wrist_quat_des = quat_mul(robot_wrist_quat, actions_rot)

            # convert back to tip poses, and send these to IK controller
            tip_pos_des, tip_quat_des = self.convert_wrist2tip(wrist_pos_des, wrist_quat_des)
            actions_pos = wrist_pos_des
            actions_rot = wrist_quat_des

            # actions_pos, actions_rot, actions_finger = self.actions[:,:3],self.actions[:,3:7], self.actions[:,7].reshape(-1,1)
            # actions_pos += robot_pos 
   
            # # print the rotation for each action and compare to the real world
            # actions_rot = quat_mul(robot_rot, actions_rot) 
            self.actions = torch.hstack([actions_pos, actions_rot, actions_finger])
        elif control_type == "default":
            self.robot_actions[:] = self.actions
        # perform physics stepping


        from rialto.franka.diff_ik_utils import  diffik_step
        from utils_frames import th_quat_from_isaac_to_real
        # use IK to convert to joint-space commands
        target_joint_delta = diffik_step(
            robot_wrist_pos,
            th_quat_from_isaac_to_real(robot_rot), # maybe transform rot
            actions_pos,
            th_quat_from_isaac_to_real(actions_rot), # maybe transform rot
            self.robot.data.ee_jacobian.clone(),
        )    
    
        robot_joints = self.robot.data.arm_dof_pos.clone() + target_joint_delta
        # print("Predicted robot joints are:", robot_joints)
        self.robot_actions[:, : self.robot.arm_num_dof] = robot_joints
            
        # offset actuator command with position offsets
        dof_pos_offset = self.robot.data.actuator_pos_offset[:, : self.robot.arm_num_dof]
        self.robot_actions[:, : self.robot.arm_num_dof] -= dof_pos_offset
        # we assume last command is tool action so don't change that
        self.robot_actions[:, -1] = self.actions[:, -1]
        # set actions into buffers
        start = time.time()
        for _ in range(self.cfg.control.decimation):
            self.robot.apply_action(self.robot_actions)

            self.sim.step(render=self.enable_render)

            # self.scene.update_buffers(self.dt)
            # self.robot.update_buffers(self.dt)

            # simulate
            self._debug_vis()
            # check that simulation is playing
            if self.sim.is_stopped():
                return
        print("All decimation took", time.time() - start)
        self.previous_robot_actions = self.robot_actions.clone()

        # post-step:
        # -- compute common buffers
        self.robot.update_buffers(self.dt)
        self.scene.update_buffers(self.dt)

        pred_joints = {}
        for i in range(len(self.robot_actions[0])-1):
            pred_joints['ik/'+str(i)+"_error"] = torch.norm(self.robot_actions[:,i] + dof_pos_offset[:,i] - self.robot.data.arm_dof_pos[:, i])
        
        error = self.robot.data.ee_state_w.clone()[:,:3] - self.envs_positions - tip_pos_des
        pred_joints["ik/x_error"] = torch.norm( error[:,0])
        pred_joints["ik/y_error"] = torch.norm( error[:,1])
        pred_joints["ik/z_error"] = torch.norm( error[:,2])

        # get action error
        wandb.log(pred_joints)
        # -- compute MDP signals
        # reward
        self.reward_buf = self._reward_manager.success_reward(self)
        # terminations
        self._check_termination()
        # -- store history
        self.previous_actions = self.actions.clone()

        # -- add information to extra if timeout occurred due to episode length
        # Note: this is used by algorithms like PPO where time-outs are handled differently
        self.extras["time_outs"] = self.episode_length_buf >= self.max_episode_length
        self.extras["success_rate"] = self._reward_manager.success_rate(self,)
        # -- add information to extra if task completed
        
        robot_data = self.robot.data.arm_dof_pos
        while torch.any(robot_data.isnan()):
            print("Robot data is nan in step")
            nanidx = torch.where(robot_data.isnan())[0][0]
            print(nanidx)
            print(robot_data_beg[nanidx])
            print(actions[nanidx])
            print(self.previous_robot_actions[nanidx])
            print(robot_data_ee_state[nanidx])
            # import wandb
            # from datetime import timedelta
            # from wandb import AlertLevel
            # wandb.alert(
            #     title='Nan data found',
            #     text=f'Nan data in step',
            #     level=AlertLevel.WARN,
            #     wait_duration=timedelta(minutes=1)
            # )
            idx = torch.tensor(copy.deepcopy(torch.unique(torch.where(robot_data.isnan().clone())[0].clone()).clone().detach().cpu().numpy()))
            print(idx)

            self._reset_idx(idx)
            # self.sim.step(render=self.enable_render)
            robot_data = self._get_observations()['policy']['tool_positions']
            print(torch.where(robot_data.isnan()))
            print(torch.where(self.robot.data.arm_dof_pos.isnan()))


        # if self.cfg.viewer.debug_vis and self.enable_render:
        #     self._debug_vis()

    def get_dense_reward(self):
        env_type = self.cfg.reward_type

        if env_type == "mug":
            assert False
            reward = self.scene.data["Object_269"].root_pos_w[:,2] > 0.1
        elif env_type == "dishsinklab":
            assert False
            site_plate = self.scene.data["Object_281"].root_pos_w
            site_dishrack = self.scene.data["Object_280"].root_pos_w
            quat_plate = self.scene.data["Object_270"].root_quat_w

            rot_mat_plate = matrix_from_quat(quat_plate)

            plate_y = -1.0 * rot_mat_plate[:, :, 1]
            z_up = torch.tensor([[0, 0, 1]]).float().to(self.device).repeat((self.num_envs, 1))
            plate_z_dot = (plate_y * z_up).sum(1)

            success_rot = ((1 - torch.abs(plate_z_dot)) < np.deg2rad(20)).squeeze()
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
           
            success_pos = torch.linalg.norm(site_dishrack - site_plate, axis=1) < 0.25
            print("distance", torch.linalg.norm(site_dishrack - site_plate, axis=1), (1 - torch.abs(plate_z_dot)))
            reward = torch.logical_and(success_pos, torch.logical_and(success_rot, success_gripper_open))
            print("success rot, success pos", success_rot, success_pos)
        elif env_type == "toynbowl" or env_type == "toycarnbowl":
            site_bowl = self.scene.data["Object_269"].root_pos_w
            site_toy = self.scene.data["Object_272"].root_pos_w
            print("cup reach", torch.linalg.norm(site_toy - site_bowl, axis=1))
            reward_done = self.get_reward()*10
            general_reward = 1 - torch.linalg.norm( site_toy - site_bowl, axis=1) - torch.linalg.norm( site_toy - self.robot.data.ee_state_w[:,:3], axis=1)
            general_reward = torch.clip(general_reward, -2, 2)
            reward = general_reward + reward_done

            print("reward", reward)
            # TODO : I should clip the general_reward so that it doesn't go too extreme
        else:
            print("Reward not specified")
            assert False

        reward = reward.long()
        return reward

    def get_reward(self):
        env_type = self.cfg.reward_type

        if env_type == "mug":
            reward = self.scene.data["Object_269"].root_pos_w[:,2] > 0.1
        elif env_type == "drawer":
            reward = self.scene.data["Object_274/PrismaticJoint"].root_state_dof[:,0] > 0.1
        elif env_type == "drawer_new":
            reward = self.scene.data["Object_266/PrismaticJoint"].root_state_dof[:,0] > 0.1
        elif env_type == "bowlnrack":
            site_bowl = self.scene.data["Object_186"].root_pos_w
            site_dishrack = self.scene.data["Object_268"].root_pos_w
            rot_bowl = self.scene.data["Object_186"].root_quat_w
            q = rot_bowl
            q_conj = quat_inv(q)
            vect = quat_mul(quat_mul(q, torch.tensor([[0,0,1,0]]).to(self.device)), q_conj)
            success_rot = vect[:,-1] < -0.2
           
            print("robot pos", torch.linalg.norm(site_dishrack - site_bowl))
            success_pos = torch.linalg.norm(site_dishrack - site_bowl) < 0.2

            reward = torch.logical_and(success_pos, success_rot)
            print("success rot", success_rot, success_pos)
        elif env_type == "dishnrack":
            site_plate = self.scene.data["Object_281"].root_pos_w
            site_dishrack = self.scene.data["Object_280"].root_pos_w
            quat_plate = self.scene.data["Object_270"].root_quat_w

            #self.plate_marker.set_world_poses(self.scene.data["Object_270"].root_state_w[:, 0:3].clone(), self.scene.data["Object_270"].root_state_w[:, 3:7].clone())
            #self.dishrack_marker.set_world_poses(self.scene.data["Object_278"].root_state_w[:, 0:3].clone(), self.scene.data["Object_278"].root_state_w[:, 3:7].clone())

            # get the orientation of the plate site
            # rot_mat_site_plate = matrix_from_quat(self.scene.data["Object_281"].root_quat_w)
            # rot_mat_site_dishrack = matrix_from_quat(self.scene.data["Object_280"].root_quat_w)
            rot_mat_plate = matrix_from_quat(quat_plate)

            # plate z and rack y should align
            # site_plate_z = rot_mat_site_plate[:, :, 2]
            # site_dishrack_y = rot_mat_site_dishrack[:, :, 1]
            # plate_rack_align_dot = torch.matmul(site_plate_z, site_dishrack_y.T)
            # plate_z = rot_mat_plate[:, :, 2]
            plate_y = -1.0 * rot_mat_plate[:, :, 1]
            z_up = torch.tensor([[0, 0, 1]]).float().to(self.device).repeat((self.num_envs, 1))
            # plate_z_dot = torch.matmul(plate_y, z_up.T)
            plate_z_dot = (plate_y * z_up).sum(1)

            # dot product should be close to 1
            # success_rot = ((1 - torch.abs(plate_rack_align_dot)) < np.deg2rad(5)).squeeze()
            success_rot = ((1 - torch.abs(plate_z_dot)) < np.deg2rad(20)).squeeze()
            # print(f'plate_z: {plate_y}, plate_z_dot: {plate_z_dot}, success_rot: {success_rot}')

            # q = rot_plate
            # q_conj = quat_inv(q)
            # vect = quat_mul(quat_mul(q, torch.tensor([[0,0,1,0]]).to(self.device)), q_conj)
            # success_rot = torch.abs(vect[:,-1]) < 0.5
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
           
            # print("robot pos", torch.linalg.norm(site_dishrack - site_plate))
            success_pos = torch.linalg.norm(site_dishrack - site_plate, axis=1) < 0.21
            print("distance", torch.linalg.norm(site_dishrack - site_plate, axis=1), (1 - torch.abs(plate_z_dot)))
            reward = torch.logical_and(success_pos, torch.logical_and(success_rot, success_gripper_open))
            print("success rot, success pos", success_rot, success_pos)
            # reward = torch.logical_and(success_pos, success_gripper_open)
            # print("success ", success_pos)
        elif env_type == "dishsinklab":
            site_plate = self.scene.data["Object_281"].root_pos_w
            site_dishrack = self.scene.data["Object_280"].root_pos_w
            quat_plate = self.scene.data["Object_270"].root_quat_w

            rot_mat_plate = matrix_from_quat(quat_plate)

            plate_y = -1.0 * rot_mat_plate[:, :, 1]
            z_up = torch.tensor([[0, 0, 1]]).float().to(self.device).repeat((self.num_envs, 1))
            plate_z_dot = (plate_y * z_up).sum(1)

            success_rot = ((1 - torch.abs(plate_z_dot)) < np.deg2rad(20)).squeeze()
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
           
            success_pos = torch.linalg.norm(site_dishrack - site_plate, axis=1) < 0.25
            print("distance", torch.linalg.norm(site_dishrack - site_plate, axis=1), (1 - torch.abs(plate_z_dot)))
            reward = torch.logical_and(success_pos, torch.logical_and(success_rot, success_gripper_open))
            print("success rot, success pos", success_rot, success_pos)
        elif env_type == "booknshelve":
            site_book = self.scene.data["Object_279"].root_pos_w
            site_shelve = self.scene.data["Object_280"].root_pos_w
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            success_reach=torch.linalg.norm(site_book - site_shelve, axis=1) < 0.06
            reward = torch.logical_and(success_reach, success_gripper_open)
        elif env_type == "cupntrash":
            site_cup = self.scene.data["Object_267"].root_pos_w
            site_trash = self.scene.data["Object_268"].root_pos_w
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            success_reach=torch.linalg.norm(site_cup - site_trash, axis=1) < 0.12
            print("cup reach", torch.linalg.norm(site_cup - site_trash, axis=1))
            reward = torch.logical_and(success_reach, success_gripper_open)
        elif env_type == "toynbowl":
            site_bowl = self.scene.data["Object_269"].root_pos_w
            site_toy = self.scene.data["Object_272"].root_pos_w
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            success_reach=torch.linalg.norm( site_toy - site_bowl, axis=1) < 0.11
            print("cup reach", torch.linalg.norm(site_toy - site_bowl, axis=1))
            reward = torch.logical_and(success_reach, success_gripper_open)
        elif env_type == "toycarnbowl":
            site_bowl = self.scene.data["Object_269"].root_pos_w
            site_toy = self.scene.data["Object_272"].root_pos_w
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            success_reach=torch.linalg.norm( site_toy - site_bowl, axis=1) < 0.11
            print("cup reach", torch.linalg.norm(site_toy - site_bowl, axis=1))
            reward = torch.logical_and(success_reach, success_gripper_open)
        elif env_type == "mugandshelf":
            site_shelve = self.scene.data["Object_271"].root_pos_w
            site_mug = self.scene.data["Object_264"].root_pos_w
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            success_reach=torch.linalg.norm(site_mug - site_shelve, axis=1) < 0.12
            rot_mug = self.scene.data["Object_263"].root_quat_w
            q = rot_mug
            q_conj = quat_inv(q)
            compare_tensor = np.tile([0,0,1,0], (self.num_envs, 1))
            vect = quat_mul(quat_mul(q, torch.from_numpy(compare_tensor).to(self.device)), q_conj)
            upright = torch.abs(vect[:,-1]) > 0.95

            print("success reach", torch.linalg.norm(site_mug - site_shelve, axis=1))
            success = torch.logical_and(torch.logical_and(success_reach, success_gripper_open), upright)
            reward = torch.zeros(self.num_envs).to(self.device)
            reward[success] = 1
            # TODO set zero reward and where True set reward to 1

        elif env_type == "mugnrack":
            site_mug = self.scene.data["Object_263"].root_pos_w
            site_rack = self.scene.data["Object_270"].root_pos_w
            success_reach = torch.linalg.norm(site_mug - site_rack, axis=1) < 0.06
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03

            reward = torch.logical_and(success_reach, success_gripper_open)
        elif env_type == "mugnrackdense":
            site_mug = self.scene.data["Object_263"].root_pos_w
            site_rack = self.scene.data["Object_270"].root_pos_w
            success_reach = torch.linalg.norm(site_mug - site_rack, axis=1) < 0.06
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03

            reward = torch.logical_and(success_reach, success_gripper_open).long() * 0.5 + success_reach.long() * 0.5
        elif env_type == "drawer_bigger":
            # TODO: add gripper open
            success_open = self.scene.data["Object_266/PrismaticJoint"].root_state_dof[:,0] > 0.12
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            reward = torch.logical_and(success_gripper_open, success_open).long()

        elif env_type == "cabinet":
            success_open = self.scene.data["Object_271/RevoluteJoint"].root_state_dof[:,0] > 0.5
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03
            reward = torch.logical_and(success_open, success_gripper_open)
        elif env_type == "mugupright":
            rot_mug = self.scene.data["Object_262"].root_quat_w
            q = rot_mug
            q_conj = quat_inv(q)
            compare_tensor = np.tile([0,0,1,0], (self.num_envs, 1))
            vect = quat_mul(quat_mul(q, torch.from_numpy(compare_tensor).to(self.device)), q_conj)
            upright = torch.abs(vect[:,-1]) > 0.98
            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03

            reward = torch.logical_and(success_gripper_open, upright)
        elif env_type == "kitchentoaster":
            print("toaster joint, ", self.scene.data["Object_270/RevoluteJoint"].root_state_dof[:,0] )
            open_toaster = self.scene.data["Object_270/RevoluteJoint"].root_state_dof[:,0] > 0.65

            success_gripper_open = self.robot.data.tool_dof_pos[:,0] > 0.03

            reward = torch.logical_and(success_gripper_open, open_toaster)

        reward = reward.long()
        return reward

    def get_success(self):
        success = self.get_reward() == 1
        return success
        
    def get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()
    def _get_observations(self) -> VecEnvObs:
        # compute observations
        return self._observation_manager.compute()

    """
    Helper functions - Scene handling.
    """

    def _pre_process_cfg(self) -> None:
        # """Pre-processing of configuration parameters."""
        # # set configuration for task-space controller
        # if self.cfg.control.control_type == "inverse_kinematics":
        #     print("Using inverse kinematics controller...")
        #     # enable jacobian computation
        #     self.cfg.robot.data_info.enable_jacobian = True
        #     # enable gravity compensation
        #     self.cfg.robot.rigid_props.disable_gravity = True
        #     # set the end-effector offsets
        #     self.cfg.control.inverse_kinematics.position_offset = self.cfg.robot.ee_info.pos_offset
        #     self.cfg.control.inverse_kinematics.rotation_offset = self.cfg.robot.ee_info.rot_offset
        # else:
        #     print("Using default joint controller...")
        pass

    def _process_cfg(self) -> None:
        """Post processing of configuration parameters."""
        # compute constants for environment
        self.dt = self.cfg.control.decimation * self.physics_dt  # control-dt
        self.max_episode_length = math.ceil(self.cfg.env.episode_length_s / self.dt)

    def _initialize_views(self) -> None:
        """Creates views and extract useful quantities from them."""
        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        self.sim.reset()
        # define views over instances
        self.robot.initialize(self.env_ns + "/.*/Robot")
        self.scene.initialize(self.env_ns + "/.*/Scene", self.envs_positions)

        # create controller
        if self.cfg.control.control_type == "inverse_kinematics":
            # self._ik_controller = DifferentialInverseKinematics(
            #     self.cfg.control.inverse_kinematics, self.robot.count, self.device
            # )
            ik_control_cfg = DifferentialInverseKinematicsCfg(
                command_type="pose_abs",
                ik_method="lstsq",
                position_offset=self.robot.cfg.ee_info.pos_offset,
                rotation_offset=self.robot.cfg.ee_info.rot_offset,
            )
            if self.ik_solver_type ==  ISAAC_PINV_SOLVER:
                self._ik_controller = DifferentialInverseKinematics(ik_control_cfg, self.num_envs, self.sim.device)
            elif self.ik_solver_type == QP_SOLVER:
                self._ik_controller = QPDifferentialInverseKinematics(ik_control_cfg, self.num_envs, self.sim.device)
            else:
                self._ik_controller = CustomPBIKSolver()
            self.num_actions = self._ik_controller.num_actions + 1
        elif self.cfg.control.control_type == "default":
            self.num_actions = self.robot.num_actions

        # history
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        self.previous_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)
        # robot joint actions
        self.robot_actions = torch.zeros((self.num_envs, self.robot.num_actions), device=self.device)

        self.scene.update_buffers(self.dt)
        self.robot.update_buffers(self.dt)

    """
    Helper functions - MDP.
    """

    def _check_termination(self) -> None:
        # extract values from buffer
        self.reset_buf[:] = 0
        # if self.cfg.terminations.episode_timeout:
        self.reset_buf = torch.where(self.episode_length_buf >= self.max_episode_length, 1, self.reset_buf)

       
    def _reset_scene_initial_pose(self, env_ids):
        """Randomize the initial pose of the object."""
        # get the default root state
        root_state = self.scene.get_default_root_state(env_ids)
    
        for object_name in root_state:
            if "joint" in object_name.lower():
                continue
            root_state[object_name][:, 0:3] += self.envs_positions[env_ids]
        
        self.scene.set_root_state(root_state, env_ids=env_ids)


class SceneObservationManager(ObservationManager):
    """Reward manager for single-arm reaching environment."""

    def arm_dof_pos(self, env: GeneralEnv):
        """DOF positions for the arm."""
        obs = env.robot.data.arm_dof_pos
        # if torch.any(obs.isnan()):
            # print("arm dof pos", obs)
        return obs

    def tool_dof_pos_scaled(self, env: GeneralEnv):
        """DOF positions of the tool normalized to its max and min ranges."""
        obs = scale_transform(
            env.robot.data.tool_dof_pos,
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 0],
            env.robot.data.soft_dof_pos_limits[:, env.robot.arm_num_dof :, 1],
        )
        # if torch.all(obs.isnan()):
        if torch.any(obs < 0):
            obs[torch.where(obs[:,1] < 0)] = torch.tensor([-1.0,-1.0]).to(self.device)
            # print("too dof pos scaled", obs)
        return obs

    def tool_positions(self, env: GeneralEnv):
        """Current end-effector position of the arm."""
        obs = env.robot.data.ee_state_w[:, :3].clone() - env.envs_positions
        # if torch.all(obs.isnan()):
        # print("too positions", obs)
        return obs.clone()

    def tool_orientations(self, env: GeneralEnv):
        """Current end-effector orientation of the arm."""
        # make the first element positive
        quat_w = env.robot.data.ee_state_w[:, 3:7].clone()
        # quat_w[quat_w[:, 0] < 0] *= -1
        # if torch.all(quat_w.isnan()):
        # print("tool orientation", quat_w)
        return quat_w.clone()

    def objects_pos(self, env: GeneralEnv):
        """Scene joint state."""
        obs = []
        for object_name in env.scene.data_objects:
            if "joint" in object_name.lower():
                pos = env.scene.data[object_name].root_pos_w[:,:3]
                obs.append(pos) 
            else:
                pos = env.scene.data[object_name].root_state_w[:,:3] + env.scene.object_coms[object_name][0] # TODO: Remove this!!
                par_obs = pos - env.envs_positions 
                obs.append(par_obs)

        obs = torch.hstack(obs)
        # if torch.all(obs.isnan()):
            # print("scene site", obs)
        return obs
    
    def objects_rot(self, env: GeneralEnv):
        """Scene joint state."""
        obs = []
        for object_name in env.scene.data_objects:
            if "joint" not in object_name.lower():
                rot = env.scene.data[object_name].root_state_w[:,3:7]
                obs.append(rot)
        obs = torch.hstack(obs)
        # if torch.all(obs.isnan()):
            # print("scene site", obs)fa
        return obs
    
    def arm_actions(self, env: GeneralEnv):
        """Last arm actions provided to env."""
        obs = env.actions[:, :-1]
        # if torch.all(obs.isnan()):
            # print("arm actions", obs)
        return obs

    def tool_actions(self, env: GeneralEnv):
        """Last tool actions provided to env."""
        obs = env.actions[:, -1].unsqueeze(1)
        # if torch.all(obs.isnan()):
            # print("tool actions", obs)
        return obs
    
    # def scene_rot(self, env:GeneralEnv):
    #     # rot = env.scene.site_bodies["Site"].get_world_poses()[1]
    #     rot = env.scene.data[list(env.scene.data.keys())[0]].root_state_w[:,3:7]

    #     # if torch.all(rot.isnan()):
    #         # print("Scene rot", rot)

    #     return rot
    
    # def scene_pos(self, env:GeneralEnv):
    #     pos = env.scene.data[list(env.scene.data.keys())[0]].root_state_w[:,:3]
    #     pos = pos - env.envs_positions

    #     return pos

class SceneRewardManager(RewardManager):
    """Reward manager for single-arm object lifting environment."""
    def simple_reaching_reward(self, env: GeneralEnv):
        # env.scene.data[list(env.scene.data.keys())[1]]
        # franka_grasp_pos = env.robot.data.ee_state_w[:, 0:3]
        # franka_grasp_pos[:,2] += 0.05
        # object_pos = env.scene.data[list(env.scene.data.keys())[1]].root_state_w[:, 0:3]

        # d = torch.norm(franka_grasp_pos - object_pos, p=2, dim=-1)
        # # dist_reward = 1.0/(1+d**2)
        # # dist_reward*=dist_reward
        # # dist_reward = torch.where(d <= 0.02, dist_reward*2, dist_reward)
        # # print("dist reward", dist_reward)
        # # return dist_reward
        return 0

    def success_rate(self, env):
        return env.get_success()

    def success_reward(self, env: GeneralEnv):
        # env.scene.data[list(env.scene.data.keys())[1]]
        # franka_grasp_pos = env.robot.data.ee_state_w[:, 0:3]
        # franka_grasp_pos[:2] += 0.05
        # object_pos = env.scene.data[list(env.scene.data.keys())[1]].root_state_w[:, 0:3]

        # d = torch.norm(franka_grasp_pos - object_pos, p=2, dim=-1)
        
        # success_reward = torch.zeros(env.num_envs).to(env.device)
        # # reward = torch.where(d > 0.0, success_reward, success_reward+1)
        # success_reward = torch.where(env.get_success(), success_reward+1, success_reward)
        if env.cfg.dense_reward:
            rew = env.get_dense_reward()
        else:
            rew = env.get_reward()

        return rew
    
        # dist_reward = 1.0/(1+d**2)
        # dist_reward*=dist_reward
        # dist_reward = torch.where(d <= 0.02, dist_reward*2, dist_reward)
        # print("dist reward", dist_reward)
        # return dist_reward
