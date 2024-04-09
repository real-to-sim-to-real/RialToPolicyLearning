from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
from rialto.envs.gymenv_wrapper import GymGoalEnvWrapper
import numpy as np
import gym
from gym import spaces
from gym.spaces import Discrete
from rialto.envs.utils.parse_cfg import parse_env_cfg
import torch
# from omni.isaac.orbit_envs.isaac_env import IsaacEnv, VecEnvIndices, VecEnvObs
import yaml
from rialto.envs.isaac_env_render import IsaacEnvRender
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import copy
import time
from omni.isaac.orbit.utils.math import quat_inv, quat_mul, random_orientation, sample_uniform, scale_transform, quat_from_euler_xyz
from render_mesh_utils import PointCloudRenderer, PointCloudRendererOnline

class IsaacIntermediateEnv():
  def __init__(self, action_repeat=1, log_per_goal=False, num_envs=1, max_path_length=35, randomize_object_name="", continuous_action_space=False, display=False, render_images=False, img_shape=(64,64),usd_name="",usd_path="", camera_pos="", num_cameras=2, euler_rot=False,sensors=["rgb"],randomize_rot=False, randomize_pos=False, randomize_action_mag=None, cfg=None):

    super().__init__()
    self._action_repeat = action_repeat
    self.log_per_goal = log_per_goal
    self.continuous_action_space = continuous_action_space
    task = "Isaac-Franka-General-v0"
    self.num_envs = num_envs
    env_cfg = parse_env_cfg(task, use_gpu=True, num_envs=num_envs)
    self.sensors = sensors
    self.usd_name = usd_name
    self.cfg = cfg
    if "usd" in usd_name:
      env_cfg.scene.meta_info.usd_path = f"{usd_path}/{usd_name}"
    else:
      env_cfg.scene.meta_info.usd_path = f"{usd_path}/{usd_name}.usdz"
    
    env_cfg.scene.distractor_paths = cfg["distractor_paths"]

    env_cfg.dense_reward = cfg["dense_reward"]

    if "reward_type" in cfg:
      env_cfg.reward_type = cfg["reward_type"]
    else:
      env_cfg.reward_type = cfg["env_type"]

    env_cfg.control.decimation = cfg["decimation"]
    env_cfg.randomization.randomize_pos = randomize_pos
    env_cfg.randomization.randomize_rot = randomize_rot
    env_cfg.randomization.floor_height = cfg["floor_height"]
    env_cfg.randomization.randomize_object_name = randomize_object_name
    env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = cfg["gpu_total_aggregate_pairs_capacity"]
    if "position_min_bound" in cfg:
      print("Fetching randomization from config")
      env_cfg.randomization.object_initial_pose.position_min_bound = cfg['position_min_bound']
      env_cfg.randomization.object_initial_pose.position_max_bound = cfg['position_max_bound']
      env_cfg.randomization.object_initial_pose.orientation_min_bound = cfg['orientation_min_bound']
      env_cfg.randomization.object_initial_pose.orientation_max_bound = cfg['orientation_max_bound']
    
    if len(cfg["distractor_paths"]) != 0:
      env_cfg.randomization.distractor_initial_pose.position_min_bound = cfg['distractor_pos_min_bound']
      env_cfg.randomization.distractor_initial_pose.position_max_bound = cfg['distractor_pos_max_bound']
      env_cfg.randomization.distractor_initial_pose.orientation_min_bound = cfg['distractor_ori_min_bound']
      env_cfg.randomization.distractor_initial_pose.orientation_max_bound = cfg['distractor_orit_max_bound']

    print("Displaying", display)
    self._env = gym.make(task, cfg=env_cfg, headless=not display) 

    self.randomize_action_mag = np.array(randomize_action_mag)
    self.render_images = render_images
    self.gen_synthetic_only = (len(cfg["sensors"])==1 and "synthetic_pcd" in cfg["sensors"])
    if render_images and not self.gen_synthetic_only:
      print("Rendering images")
      # sensors = ["rgb", "pointcloud"]
      self.env_render = IsaacEnvRender(task, 
            self._env, 
            first_person=False,
            sensor = self.sensors,
            num_cameras=num_cameras,
            cfg=cfg,
            )
      self.env_render.set_resolution(img_shape[1], img_shape[0])
      self.num_cameras = num_cameras

    if "synthetic_pcd" in cfg["sensors"]:
      self.synthetic_pcd = PointCloudRendererOnline(self, cfg)
    self.euler_rot = euler_rot
    self._observation_space = self._env.observation_space
    self._goal_space = self._env.observation_space
    print("observation space in kitchen", self._observation_space)
            
    if self.euler_rot:
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
                                  [[0,0,0,1,0,0,0],
                                  [0,0,0,-1,0,0,0],
                                  [0,0,0,0,1,0,0],
                                  [0,0,0,0,-1,0,0],
                                  [0,0,0,0,0,1,0],
                                  [0,0,0,0,0,-1,0],
                                  ])
      self.gripper_actions = np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,0,-1]])
    else:
      self.base_movement_actions = np.array(
                                  [[1,0,0,0,0,0,0,0],
                                  [-1,0,0,0,0,0,0,0],
                                  [0,1,0,0,0,0,0,0],
                                  [0,-1,0,0,0,0,0,0],
                                  [0,0,1,0,0,0,0,0],
                                  [0,0,-1,0,0,0,0,0]
                                  ])

      self.base_rotation_actions = np.array(
                                  [[0,0,0,1,0,0,0,0],
                                  [0,0,0,-1,0,0,0,0],
                                  [0,0,0,0,1,0,0,0],
                                  [0,0,0,0,-1,0,0,0],
                                  [0,0,0,0,0,1,0,0],
                                  [0,0,0,0,0,-1,0,0],
                                  [0,0,0,0,0,0,1,0],
                                  [0,0,0,0,0,0,-1,0]
                                  ])
      
      self.gripper_actions = np.array([[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,-1]])

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

    self.initial_obs = np.zeros((self.num_envs, self._observation_space.shape[0]))
    self.initial_obs = self.reset()['observation']

    print("initial obs", self.initial_obs)
    print("Observation space", self._observation_space)

  def generate_goal(self,):
       
    print("There are no goals in IsaacSimEnv")

    return np.zeros((self.initial_obs.shape))

  def treat_pcd(self, pcd, to_robot_frame):

    all_pcds_points = []
    all_pcds_colors = []
    robot_frame = self._env.get_robot_frame().detach().cpu().numpy().copy()
    
    for i in range(self.num_envs):

      pcd_joint_points = np.array([])
      pcd_joint_colors = np.array([])
    
      for pcd_idx in range(self.num_cameras):
        points, colors = np.asarray(pcd[i*self.num_cameras + pcd_idx])

        if to_robot_frame:
          # import IPython
          # IPython.embed()
          robot_pos = robot_frame[i][:3]
          robot_quat = robot_frame[i][3:7]
          mid = robot_quat[0]
          robot_quat[:3] = robot_quat[1:]
          robot_quat[-1] = mid
          rot_matrix = R.from_quat(robot_quat).as_matrix()
          # new_points = points - robot_pos
          points = torch.tensor(points).to("cuda:0").float()
          rot_matrix = torch.tensor(rot_matrix).to("cuda:0").float()
          rot_matrix = torch.tensor(rot_matrix).to("cuda:0").float()
          robot_pos = torch.tensor(robot_pos).to("cuda:0").float()
          new_points = torch.matmul(points, rot_matrix)
          points = new_points - torch.matmul(robot_pos, rot_matrix)

          crop_max = 2
          crop_min = -2
          
          valid = points < crop_max
          valid = torch.logical_and(points > crop_min, valid)
          valid_new = torch.logical_and(valid[:,0], valid[:,1])
          valid_new = torch.logical_and(valid_new, valid[:,2])

          points = points[valid_new].cpu().numpy()
          # # colors = colors[valid_new]
          # import IPython
          # IPython.embed()
          
          # new_pcd = o3d.geometry.PointCloud()
          # new_pcd.points = o3d.utility.Vector3dVector(points)
          # x  = np.linspace(-1, 1, 100)
          # y  = np.linspace(-1, 1, 100)
          # xv, yv = np.meshgrid(x, y)
          # xv = xv.flatten()
          # yv = yv.flatten()
          # zs = np.zeros_like(xv)
          # pcd_plane_points = np.stack([xv, yv, zs]).T
          # pcd_plane = o3d.geometry.PointCloud()
          # pcd_plane.points = o3d.utility.Vector3dVector(pcd_plane_points)
          # o3d.visualization.draw_geometries([new_pcd, pcd_plane])

        if len(pcd_joint_points) == 0:
          pcd_joint_points = points
          # pcd_joint_colors = points
        else:
          pcd_joint_points = np.concatenate((pcd_joint_points, points), axis=0)
          # pcd_joint_colors = np.concatenate((pcd_joint_colors, colors), axis=0)
        # pcd_joint.points = o3d.utility.Vector3dVector(points)
        # pcd_joint.colors = o3d.utility.Vector3dVector(colors)
      
      all_pcds_points.append(points)
      all_pcds_colors.append(np.ones_like(points))
    return all_pcds_points, all_pcds_colors

  def render_image(self, sensors=["rgb"], combined_pcd=True, to_robot_frame=True):
      # sensors = ["rgb", "pointcloud"]
      if self.gen_synthetic_only:
        if "synthetic_pcd" in self.cfg["sensors"]:
          obs = self._get_obs(self._env.get_observations())['observation']
          synthetic_pcd = self.synthetic_pcd.generate_pcd()

          return np.zeros((obs.shape[0],64,64,3)), (synthetic_pcd, synthetic_pcd)
      if self.render_images:
        obs_image = self.env_render.render(sensors)
        if 'rgb' in sensors:
          pre_image = np.array(obs_image['rgb'])[:,:,:,:3, 0]
          image_array = pre_image#np.concatenate(pre_image)
        else:
          image_array = np.zeros((self.num_envs,64,64,3))
          
        if "pointcloud" in sensors or "distance_to_image_plane" in obs_image:
            if "pointcloud" in sensors:
              pcd = obs_image["pointcloud"]
            else:
              pcd = obs_image["distance_to_image_plane"]
              image_array = np.array(obs_image["depth_img"])
              image_array[np.isinf(image_array)] = 0
              image_array[image_array > 4] = 0
              image_array = np.expand_dims(image_array, axis=-1)
              image_array = (255 * np.ones((*image_array.shape[:-1], 3)) * image_array / np.max(image_array)).astype(np.int32)
            # import matplotlib.pyplot as plt
            # plt.imshow(image_array[0])
            # plt.show()
            all_pcds_points, all_pcds_colors =  self.treat_pcd(pcd, to_robot_frame)
            return image_array, (all_pcds_points, all_pcds_colors)
        else:
          # images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
          
          # wandb.log({"cameraviews": images})
          #obs_image = env_render.render(["semantic_segmentation"])
          # seg_image  = np.array(obs_image['semantic_segmentation'])[:,:,:,0]
          return image_array, (image_array,image_array)
      else:
        return np.zeros((64,64,3)), (np.zeros((self.num_envs, 64*64,3)),np.zeros((self.num_envs, 64*64,3)))

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

    observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.state_space),
        ])
    return observation_space
  
  def _get_obs(self, state):
    # TODO rewrite
    #image = self._env.render('rgb_array', width=self._env.imwidth, height =self._env.imheight)
    #obs = {'image': image, 'state': state, 'image_goal': self.render_goal(), 'goal': self.goal}'

    state = torch.hstack(list(state['policy'].values())).detach().cpu().numpy()
    world_obs = state
    # ee_quat = self._env.get_ee_quat()
    # ee_obs = self._env.get_ee_pose()

    obs = world_obs#np.concatenate([world_obs, task_success, ee_quat,  ee_obs])
    goal = self.goal #self._env.goal

    return dict(
            observation=obs,
            desired_goal=goal,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=goal,
            state_achieved_goal=obs
    )


  def step(self, action, continuous_action=False):
    start = time.time()
    # TODO: Make sure step action is the same on the isaac sim envs
    total_reward = 0.0
    if self.continuous_action_space or continuous_action:
      cont_action = action
      # unnorm_cont_action = cont_action.copy()
      # cont_action[:,:3] *= 0.03
      # cont_action[:,3:] *= 0.2
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
      # unnorm_cont_action = cont_action.copy()
      cont_action[:,:3] *= 0.03
      cont_action[:,3:] *= 0.2

    if self.randomize_action_mag is not None:
      cont_action[:,:6] += np.random.uniform(-self.randomize_action_mag, self.randomize_action_mag, (cont_action.shape[0], 6))

    if self.euler_rot:
      action_ee_pos = cont_action[:,:3]
      action_ee_rot_euler = cont_action[:,3:6]
      action_ee_gripper = cont_action[:,-1].reshape(-1,1)
      action_ee_rot_quat = quat_from_euler_xyz(action_ee_rot_euler[:,0],action_ee_rot_euler[:,1],action_ee_rot_euler[:,2]) #torch.tensor().reshape(-1,4))
      cont_action = torch.hstack([action_ee_pos, action_ee_rot_quat, action_ee_gripper])
    
    repeat_actions = self._action_repeat
    if continuous_action:
      repeat_actions = 30
    for step in range(repeat_actions):
      if continuous_action:
        state, reward, done, info = self._env.step(cont_action, control_type="default")
      else:
        state, reward, done, info = self._env.step(cont_action)
      # reward = #0 #self.compute_reward()
      total_reward += reward
      if torch.any(done):
        break
    obs = self._get_obs(state)
    for k, v in obs.items():
      if 'metric_' in k:
        info[k] = v

    info['robot_joints'] = self._env.get_robot_joints()
    info['cont_action'] = cont_action
    print("All step took", time.time() - start)
    return obs, total_reward, done, info

  def reset(self):
    if self.render_images and not self.gen_synthetic_only:
      self.env_render.reset()
    state = self._env.reset()

    self.gripper_joint = np.zeros(self.num_envs)
    self.goal = self.generate_goal()#self.goals[self.goal_idx]

    return self._get_obs(state)

class IsaacGoalEnv(GymGoalEnvWrapper):
    def __init__(self, max_path_length=50, img_shape=(64,64), continuous_action_space=False,randomize_object_name="",  display=False, render_images=False, usd_name="", usd_path="", num_envs=1, sensors=["rgb"], num_cameras=1,euler_rot=False, randomize_rot=False, randomize_pos=False, randomize_action_mag=None, cfg=None):

        env = IsaacIntermediateEnv(continuous_action_space=continuous_action_space, randomize_object_name=randomize_object_name, display=display, render_images=render_images, max_path_length=max_path_length, img_shape=img_shape, usd_path=usd_path,usd_name=usd_name,num_envs=num_envs, randomize_action_mag=randomize_action_mag, sensors=sensors, num_cameras=num_cameras, euler_rot=euler_rot, randomize_pos=randomize_pos, randomize_rot=randomize_rot, cfg=cfg)
       

        super(IsaacGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal',max_path_length=max_path_length
        )

        self.action_low = np.array([0.25, -0.5])
        self.action_high = np.array([0.75, 0.5])

        self.continuous_action_space = continuous_action_space

        self.usd_name = usd_name

    def compute_success(self, achieved_state, goal):
      return 0
      final_state = achieved_state[:,-1]
      succ = []
      for fs in final_state:
        succ.append(self.base_env.compute_success(final_state, goal))

      return np.mean(succ)
      if "mug" in self.usd_name:
        print("hi")
      return self.compute_shaped_distance(achieved_state, goal) < 0.05
      #return int(per_obj_success['slide_cabinet'])  + #int(per_obj_success['hinge_cabinet'])+ int(per_obj_success['microwave'])
    
    def plot_trajectories(self,obs=None, goal=None, filename=""):
       return

    # The task is to open the microwave, then open the slider and then open the cabinet
    def compute_shaped_distance(self, achieved_state, goal):
        return np.linalg.norm(achieved_state[6:9] - achieved_state[-3:])

    def render_image(self, sensors=["rgb"]):
      return self.base_env.render_image(sensors=sensors)
    
    def get_diagnostics(self, trajectories, desired_goal_states):
 
        return OrderedDict()