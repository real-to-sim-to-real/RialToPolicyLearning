from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from scipy.spatial.transform import Rotation as R
import torch
# from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
import yaml
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
from rialto.franka.robot_env_diff_ik import RobotEnv

class RealFrankaEnv():
  def __init__(self, action_repeat=1, background_loop=False, log_per_goal=False, max_path_length=35, hz=1, continuous_action_space=False, display=False, img_shape=(64,64),usd_name="",usd_path="", camera_pos="", num_cameras=2, euler_rot=True,sensors=["rgb"], camera_pos_rand=None, camera_target=None, camera_target_rand=None, randomize_rot=False, cam_index=2, randomize_pos=False, cfg=None):

    super().__init__()
    self._action_repeat = action_repeat
    self.log_per_goal = log_per_goal
    self.continuous_action_space = continuous_action_space
    self.cam_index = cam_index
    print("Cam index", cam_index)
    self._env = RobotEnv( DoF=7,
                # ip_address="173.16.0.1",
                ip_address="173.16.0.1",
                ee_pos=True,
                qpos=True,
                flat_obs=True,
                normalize_obs=False,
                max_path_length=30,
                goal_state='right_closed',
                randomize_ee_on_reset=False,
                seed=0,
                video=True,
                sim=False,
                use_gripper=True,
                cam_idx=cam_index,
                background_loop=background_loop,
                gripper_force=cfg["gripper_force"],
                hz=hz,
                cfg=cfg)
    self.hz = hz
    self.euler_rot = euler_rot
    self._observation_space = self._env.observation_space
    self._goal_space = self._env.observation_space
    print("observation space in kitchen", self._observation_space)
            
    # if self.euler_rot:
    print("Using euler rotation")
    self.base_movement_actions = np.array(
                              [[1,0,0,0,0,0,0],
                              [-1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,-1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,-1,0,0,0,0]
                              ])

    self.base_rotation_actions = np.array(
                                [[0,0,0,1,0,0,0], # inverted with respect to sim
                                [0,0,0,-1,0,0,0], 
                                [0,0,0,0,1,0,0],
                                [0,0,0,0,-1,0,0],
                                [0,0,0,0,0,1,0], # inverted with respect to sim
                                [0,0,0,0,0,-1,0],
                                ])
    self.gripper_actions = np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1]])
    # else:
    #   self.base_movement_actions = np.array(
    #                               [[1,0,0,0,0,0,0,0],
    #                               [-1,0,0,0,0,0,0,0],
    #                               [0,1,0,0,0,0,0,0],
    #                               [0,-1,0,0,0,0,0,0],
    #                               [0,0,1,0,0,0,0,0],
    #                               [0,0,-1,0,0,0,0,0]
    #                               ])

    #   self.base_rotation_actions = np.array(
    #                               [[0,0,0,1,0,0,0,0],
    #                               [0,0,0,-1,0,0,0,0],
    #                               [0,0,0,0,1,0,0,0],
    #                               [0,0,0,0,-1,0,0,0],
    #                               [0,0,0,0,0,1,0,0],
    #                               [0,0,0,0,0,-1,0,0],
    #                               [0,0,0,0,0,0,1,0],
    #                               [0,0,0,0,0,0,-1,0]
    #                               ])
      
      # self.gripper_actions = np.array([[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,-1]])

    self.base_actions = np.concatenate([self.base_movement_actions, self.base_rotation_actions, self.gripper_actions])
    self.gripper_joint = 0
    if not self.continuous_action_space:
      if self.euler_rot:
        self._action_space = Discrete(14)
      else:
        self._action_space = Discrete(16)

      self.num_actions = self._action_space.n
    else:
      self._action_space = self._env.action_space
    print("Action space", self._action_space)
    self.initial_obs = np.zeros((1, self._observation_space.shape[0]))
    self.initial_obs = self.reset()

    print("initial obs", self.initial_obs)
    print("Observation space", self._observation_space)

  def generate_goal(self,):
    print("There are no goals in IsaacSimEnv")

    return np.zeros((self.initial_obs.shape))

  
  def render_image(self, sensors=["rgb"], combined_pcd=True, to_robot_frame=True):
    # sensors = ["rgb", "pointcloud"]
    rgb, pcd = self._env.get_rgb(), self._env.get_pcd()
    rgb = rgb[None, ...]
    pcd_points, pcd_colors = pcd
    pcd_points = pcd_points[None, ...]
    pcd_colors = pcd_colors[None, ...]
    pcd = (pcd_points, pcd_colors)
    return rgb, pcd

  def render(self, mode='rgb_array', width=480, height=64, camera_id=0):
    # TODO: fix render
    return self.render_image()
   
  @property
  def state_space(self):
    #shape = self._size + (p.linalg.norm(state - goal) < self.goal_threshold
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})
    return self._goal_space
  @property
  def action_space(self):
    return self._action_space

  @property
  def goal_space(self):
    return self._goal_space
  @property
  def observation_space(self):
    #shape = self._size + (3,)
    #space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    #return gym.spaces.Dict({'image': space})

    return self.state_space
  
  def _get_obs(self, state):
    # TODO rewrite
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'
    
    state = state["ee_pose"] #np.concatenate([state['ee_pose'], state['joints']])
    world_obs = state
    # ee_quat = self._env.get_ee_quat()
    # ee_obs = self._env.get_ee_pose()

    obs = world_obs#np.concatenate([world_obs, task_success, ee_quat,  ee_obs])
    goal = self.goal #self._env.goal

    return torch.tensor(state).to("cuda").float()

  def step(self, action):
    print("action", action)
    # TODO: Make sure step action is the same on the isaac sim envs
    total_reward = 0.0
    if self.continuous_action_space:
       cont_action = action
    else:
      cont_action = self.base_actions[action]
      # if action < 6:
      #   cont_action = self.base_movement_actions[action]
      # elif action < 14 :
      #   cont_action = self.base_rotation_actions[action - 6]
      # elif action < 16:
      #   if action == 15:
      #     self.gripper_joint = -1
      #   else:
      #     self.gripper_joint = 1
      self.gripper_joint = np.where(np.logical_or(action == self.num_actions-1, action == self.num_actions-2), -((action - (self.num_actions-2))*2 -1), self.gripper_joint)
    cont_action[:,-1] = self.gripper_joint
    cont_action = torch.Tensor(cont_action)
    if self.hz > 2:
      cont_action[:,:3] *= 0.03 
      cont_action[:,3:] *= 0.2 
    else:
      cont_action[:,:3] *= 0.03
      cont_action[:,3:] *= 0.2

    if self.euler_rot:
      action_ee_pos = cont_action[:,:3]
      action_ee_rot_euler = cont_action[:,3:6]
      action_ee_gripper = cont_action[:,-1].reshape(-1,1)
      action_ee_rot_quat = torch.tensor(R.from_euler('xyz', action_ee_rot_euler[:,:3], degrees=False).as_quat() ).reshape(-1,4)#torch.tensor().reshape(-1,4))
      cont_action = torch.hstack([action_ee_pos, action_ee_rot_quat, action_ee_gripper])
    for step in range(self._action_repeat):
      state, reward, done, info = self._env.step(cont_action)
      reward = 0 #self.compute_reward()
      total_reward += reward
      if done:
        break
    obs = self._get_obs(state)

    info['robot_joints'] = self._env.get_robot_joints()
    return obs, total_reward, done, info

  def get_robot_joints(self):
    return self._env.get_robot_joints()
    
  def reset(self):
    state = self._env.reset()

    self.gripper_joint = np.zeros(1)
    self.goal = self.generate_goal()#self.goals[self.goal_idx]

    return self._get_obs(state)
