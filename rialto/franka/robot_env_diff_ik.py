'''
Basic Robot Environment Wrapper
Robot Specific Functions: self._update_pose(), self.get_ee_pos(), self.get_ee_angle()
Camera Specific Functions: self.render_obs()
Experiment Specific Functions: self.get_info(), self.get_reward(), self.get_observation()
'''
import numpy as np
import time
import gym
import torch
import open3d as o3d
import copy
import threading
from gym.spaces import Box, Dict
import cv2
from transforms3d.euler import quat2euler, euler2quat
from transforms3d.quaternions import qmult, quat2mat
import wandb
from rialto.franka.realsense_camera import RealSenseCamera
from polymetis import RobotInterface
from polymetis import GripperInterface
from scipy.spatial.transform import Rotation as R
from rdt.common import util, path_util
from rdt.common.franka_ik import FrankaIK
from rdt.polymetis_robot_utils.plan_exec_util import PlanningHelper
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.traj_util import PolymetisTrajectoryUtil
import meshcat
import trimesh
import os, os.path as osp
from rialto.franka.diff_ik_utils import  diffik_step

poly_util = PolymetisHelper()

class RobotEnv(gym.Env):
    '''
    Main interface to interact with the robot.
    '''
    def __init__(self,
                 # control frequency
                 hz=10,
                 DoF=3,
                 # randomize arm position on reset  
                 randomize_ee_on_reset=False,
                 # allows user to pause to reset reset of the environment
                 pause_after_reset=False,
                 # observation space configuration
                 front_camera=False, 
                 side_camera=False,
                 depth_camera=False,
                 img_height=128,
                 img_width=128,
                 qpos=False,
                 ee_pos=False,
                 sphere_pos=False,
                 sphere_vel=False,
                 normalize_obs=False,
                 flat_obs=False,
                 gripper_force=100,
                 # pass IP if not running on NUC
                 ip_address=None,
                 # for state only experiments
                 goal_state=None,
                 # specify path length if resetting after a fixed length
                 max_path_length=None,
                 # use local cameras, else use images from NUC
                 local_cameras=True,
                 sim=False,
                 has_renderer=False,
                 has_offscreen_renderer=True,
                 use_gripper=False,
                 video=False,
                 cam_idx=2,
                 background_loop=False,
                 cfg=None,
                 **kwargs
                 ):

        # initialize gym environment
        super().__init__()
        
        self.times_failed = 0

        self.is_closing = -1
        # physics
        self.max_lin_vel = 0.1 # 0.9 0.4
        self.max_rot_vel = 1.5
        self.DoF = DoF
        self.hz = hz

        self._episode_count = 0
        self._max_path_length = max_path_length
        self._max_episode_steps = max_path_length
        self._curr_path_length = 0

        # resetting configuration
        self._randomize_ee_on_reset = randomize_ee_on_reset
        self._pause_after_reset = pause_after_reset
        self._gripper_angle = 1.544
        self.gripper_force = gripper_force

        self.total_loop_time = cfg["total_loop_time"] #0.75 # 0.5
        self.forward_pass_time = 0.13

        # values good with default server gains
        # self.interp_steps = 50
        # self.start_interp_offset = 20
        
        # trying with our gains (higher) 
        self.interp_steps = cfg["interp_steps"] #100    
        self.start_interp_offset = cfg["start_interp_offset"] #10

        print("interpstep", self.interp_steps, self.total_loop_time, self.start_interp_offset)
        self.cfg = cfg

        # for pushing
        # self._reset_joint_qpos = np.array([0, 0.423, 0, -1.944, 0.013, 2.219, self._gripper_angle])
        
        # self.Kx = torch.Tensor([700., 700., 700.,  65.,  65.,  65.])
        # self.Kxd = torch.Tensor([35., 35., 35.,  8.,  8.,  8.])
        # self.Kx = torch.Tensor([1000., 1000., 1000.,  90.,  90.,  90.])
        # self.Kxd = torch.Tensor([100., 100., 100.,  32.,  32.,  32.])
        self.Kx = torch.Tensor([500., 500., 500.,  35.,  35.,  35.])
        self.Kxd = torch.Tensor([40., 40., 40.,  6.,  6.,  6.])
        
        # observation space config
        self._flat_obs = flat_obs
        self._normalize_obs = normalize_obs
        
        self._front_camera = front_camera
        self._side_camera = side_camera
        self._img_height = img_height
        self._img_width = img_width
        self.depth_camera = depth_camera

        self._qpos = qpos
        self._ee_pos = ee_pos
        self._sphere_pos = sphere_pos
        self._sphere_vel = sphere_vel

        # action space
        self.use_gripper = use_gripper
        if self.use_gripper:
            self.action_space = Box(
                np.array([-1] * (self.DoF + 1)), # dx_low, dy_low, dz_low, dgripper_low
                np.array([ 1] * (self.DoF + 1)), # dx_high, dy_high, dz_high, dgripper_high
            )
        else:
            self.action_space = Box(
                np.array([-1] * (self.DoF)), # dx_low, dy_low, dz_low
                np.array([ 1] * (self.DoF)), # dx_high, dy_high, dz_high
            )

        # EE position (x, y, z) + gripper width
        self.ee_space = Box(
            np.array([0.38, -0.25, 0.00]),
            np.array([0.70, 0.28, 0.085]),
        )
    
        self.background_loop = background_loop

        # joint limits + gripper
        self._jointmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0045], dtype=np.float32)
        self._jointmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.085], dtype=np.float32)
        # self.reset_joints = np.array([-0.5030,  0.3139, -0.0901, -2.5986, -0.0178,  2.9405,  0.9503])
                # "panda_joint1": ,
        #     "panda_joint2": ,
        #     "panda_joint3": 0.0,
        #     "panda_joint4": -2.810,
        #     "panda_joint5": 0.0,
        #     "panda_joint6": 3.037,
        #     "panda_joint7": 0.741,
        # self.reset_joints = np.array([-0.5030,  0.3139, -0.0901, -2.5986, -0.0178,  2.9405,  0.9503])
        self.reset_joints = np.array([0.0,  -0.569, 0.0, -2.810, 0.0,  3.037,  0.741])
        
        # joint space + gripper
        self.qpos_space = Box(
            self._jointmin,
            self._jointmax
        )

        # final observation space configuration
        env_obs_spaces = {
            'front_camera_obs': Box(0, 255, (100, 100, 3), np.uint8),
            'side_camera_obs': Box(0, 255, (100, 100, 3), np.uint8),
            'lowdim_ee': self.ee_space,
            'lowdim_qpos': self.qpos_space,
        }
        if not self._front_camera:
            env_obs_spaces.pop('front_camera_obs', None)
        if not self._side_camera:
            env_obs_spaces.pop('side_camera_obs', None)
        if not self._qpos:
            env_obs_spaces.pop('lowdim_qpos', None)
        if not self._ee_pos:
            env_obs_spaces.pop('lowdim_ee', None)

        observation_len = 0
        if self._qpos or self._ee_pos:
            if self._qpos:
                observation_len += self.qpos_space.shape[0]
            if self._ee_pos:
                observation_len += self.ee_space.shape[0]
            if self._sphere_pos:
                observation_len += 3
            if self._sphere_vel:
                observation_len += 3
            self.observation_space = Box(-1., 1., (observation_len,), np.float32)
        if self._front_camera or self._side_camera:
            if self._front_camera:
                observation_len += 3
            if self._side_camera:
                observation_len += 3
            self.observation_space = Box(0., 255., (self._img_height, self._img_width, observation_len), np.float32)
        

        self.video = video
        self.record = None
        
        # robot configuration
        self._use_local_cameras = local_cameras
        self.sim = sim
        # from server.robot_interface import RobotInterface
        self.mc_vis = meshcat.Visualizer()
        print("Interface here")
        self._robot = RobotInterface(ip_address=ip_address)
        self._gripper = GripperInterface(ip_address=ip_address)


        self.traj_helper = PolymetisTrajectoryUtil(robot=self._robot)
        n_med = 500
        self.traj_helper.set_diffik_lookahead(int(n_med * 7.0 / 100))

        self.ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], no_gripper=True, mc_vis=self.mc_vis)
        
        tmp_obstacle_dir = osp.join(path_util.get_rdt_obj_descriptions(), 'tmp_planning_obs')
        util.safe_makedirs(tmp_obstacle_dir)
        table_obs = trimesh.creation.box([0.77, 1.22, 0.001]) #.apply_transform(util.matrix_from_list([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0]))
        table_obs_fname = osp.join(tmp_obstacle_dir, 'table.obj')
        self.ik_helper.register_object(
                    table_obs_fname,
                    pos=[0.15 + 0.77/2.0, 0.0, 0.0015],
                    ori=[0, 0, 0, 1],
                    name='table')

        self.planning = PlanningHelper(
                    mc_vis=self.mc_vis,
                    robot=self._robot,
                    gripper=self._gripper,
                    ik_helper=self.ik_helper,
                    traj_helper=self.traj_helper,
                    tmp_obstacle_dir=tmp_obstacle_dir
                )

        # self.pre_reset = np.array([-0.4419, -0.1488,  0.5121, -1.8348,  0.0666,  1.6663,  1.6654])
        # self._robot.move_to_joint_positions(np.array(self.pre_reset))
        # self._robot.move_to_joint_positions(np.array(self.reset_joints))
        self.starting_pose = self._robot.get_ee_pose()

        self._robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
        # self._robot.start_cartesian_impedance()


        # if self._use_local_cameras:
            # from camera_utils.realsense_camera import gather_realsense_cameras
            # from camera_utils.multi_camera_wrapper import MultiCameraWrapper
            # cameras = gather_realsense_cameras()
            # self._camera_reader = MultiCameraWrapper(specific_cameras=cameras)
        self.camera = RealSenseCamera(cam_idx=[cam_idx])

        # self.env_lower_limits = [0.1, -0.5, 0.155]
        # self.env_upper_limits = [0.9,  0.38, 1]
        self.env_lower_limits = [-2, -2, -2]
        self.env_upper_limits = [2, 2 , 2]
        print(f'Here before reset')
        self.reset()
        
        print(f'Here before creating control thread')
        if self.background_loop:
            self.target_joints = self._robot.get_joint_positions()
            self.current_pos = self._robot.get_joint_positions()
            self.interp_joint_pos = []
            


            self.new_target = False
            self.target_lock = threading.Lock()
            self.control_loop_thread = threading.Thread(target=self.joint_control_thread)
            self.control_loop_thread.daemon = True

            print(f'Here before starting control thread')
            self.control_loop_thread.start()

        print(f'Init done!')

    def compute_camera_intrinsic(self, fx, fy, ppx, ppy):
        return np.array([[fx, 0., ppx],
                        [0., fy, ppy],
                        [0., 0.,  1.]])

    def compute_camera_extrinsic(self, ori_mat, pos):
        cam_mat = np.eye(4)
        cam_mat[:3, :3] = ori_mat
        cam_mat[:3, 3] = pos.flatten()        
        return cam_mat
        
    def step(self, action):
        start_time = time.time()

        current_pos, current_quat = self._robot.get_ee_pose()
        delta_pos = action[0,:3]
        delta_quat = action[0,3:7]
        gripper = action[0,7]

        # import IPython
        # IPython.embed()
        # from utils_vis import meshcat_frame_show
        # import meshcat
        # vis = meshcat.Visualizer()
        # transform_sim = np.eye(4)
        # transform_sim[0:3,0:3] = quat2mat([0.0086, 0.9216, 0.0205, 0.3875])
        # transform_sim[0:3, 3] = [0.463, 0, 0.385] # current_pos
        # meshcat_frame_show(vis, "pose_sim2", transform_sim)

        # transform_real = np.eye(4)
        # transform_real[0:3,0:3] = R.from_quat(current_quat).as_matrix()@R.from_euler('z', -45, degrees=True).as_matrix()
        # transform_real[0:3, 3] = current_pos # current_pos
        # meshcat_frame_show(vis, "pose_real2", transform_real)

        
        # transform_desired = np.eye(4)
        # transform_desired[0:3,0:3] = R.from_quat(desired_quat).as_matrix()
        # transform_desired[0:3, 3] = desired_pos # current_pos
        # meshcat_frame_show(vis, "pose_desired", transform_desired)
            
        # transform_desired = np.eye(4)
        # transform_desired[0:3,0:3] = R.from_quat(desired_quat).as_matrix()@R.from_euler('z', -45, degrees=True).as_matrix()
        # transform_desired[0:3, 3] = desired_pos # current_pos
        # meshcat_frame_show(vis, "pose_desired_right_frame", transform_desired)
        # desired_quat = R.from_quat(current_quat).mult( R.from_quat(delta_quat))
        # desired_quat = (R.from_quat(current_quat) * R.from_quat(delta_quat)).as_quat() #qmult(current_quat, delta_quat) 

        updated_quat = R.from_quat(current_quat).as_matrix()@R.from_euler('z', -45, degrees=True).as_matrix()
        desired_quat_trans = updated_quat@R.from_quat(delta_quat).as_matrix()
        desired_quat = R.from_matrix(desired_quat_trans @ R.from_euler('z', 45, degrees=True).as_matrix()).as_quat()

        desired_pos = current_pos + delta_pos

        desired_pos = np.clip(desired_pos, self.env_lower_limits, self.env_upper_limits)
        
        desired_quat = torch.tensor(desired_quat)
        # TODO: set limits for the arm
        # TODO: add this when stopped testing code

        if self.background_loop:
            self.move_to_ee_pose_background(desired_pos, desired_quat)
        else:
            try:
                self.move_to_ee_pose(desired_pos,desired_quat)
            except:
                print("Error in controller!!")
                self._robot.start_joint_impedance(Kx=self.Kx, Kxd=self.Kxd)
                print("Restarted joint impedance controller")
 
        if gripper < 0 and self.gripper_open():
            self._gripper.grasp(0.07, self.gripper_force, blocking=False)
            self.is_closing = 0 
            time.sleep(1.5)

        # if self.is_closing >= 0 and self.is_closing < 1:
        #     print("Is closing here", self.is_closing)
        #     self._gripper.grasp(0.07, 10, blocking=False)
        #     self.is_closing += 1
        #     time.sleep(1.5)

        if gripper > 0 and not self.gripper_open():
            self.is_closing = -1
            self._gripper.goto(0.08, 0.05, 0.1, blocking=False)
            
            time.sleep(1)

        comp_time = time.time() - start_time
        
        reward = 0.
        self._curr_path_length += 1
        done = False
        obs_time = time.time()

        obs = self.get_observation()

        return obs, reward, done, {}

    def move_to_ee_pose(self, desired_pos, desired_quat ):
        # Interpolate a few frames
        # get joint trajectory (diffiktraj) + execute
        # desired_quat = torch.tensor(desired_quat)
        # pose_mat_des = poly_util.polypose2mat((desired_pos, torch.tensor(desired_quat)))

        # ee_poses_interpolated = self.planning.get_diffik_traj(pose_mat_des,from_current=True, N=10)
        # # print("TODO: simplify the joint diffik and simply do what we do in sim, even use the same class")
        
        # target_joint_orig = self.traj_helper.diffik_traj(ee_poses_interpolated, total_time=0.5, precompute=True, execute=False )
        
        current_pos, current_rot = self._robot.get_ee_pose()
        # current_ee_pose_mat = polypose2mat(current_ee_pose)

        current_joint_pos = self._robot.get_joint_positions()

        jacobian = self.traj_helper.robot_model.compute_jacobian(current_joint_pos)

        target_joint_delta = diffik_step(current_pos.to("cuda"), current_rot.to("cuda"), desired_pos.to("cuda"), desired_quat.to("cuda"), jacobian.to("cuda"))
        target_joints = current_joint_pos.to("cuda") + target_joint_delta

        self.interp_steps = 50
        interp_joint_pos = torch.from_numpy(np.linspace(current_joint_pos.cpu(), target_joints.squeeze().cpu(), self.interp_steps)[:]).float()
        delay = 0.5 / interp_joint_pos.shape[0]
        for j, jnt in enumerate(interp_joint_pos):
            self._robot.update_desired_joint_positions(jnt)
            if j < (interp_joint_pos.shape[0] - 1):
                time.sleep(delay)

        if False:
            if self.hz > 2:
                self._robot.update_desired_joint_positions(target_joints[0])
            else:
                self._robot.move_to_joint_positions(target_joints[0], time_to_go=1.0/self.hz)

        reached_pose = self._robot.get_ee_pose()
        reached_joints = self._robot.get_joint_positions()

        pred_joints = {}
        for i in range(len(reached_joints)):
            pred_joints['ik/'+str(i)+"_error"] = torch.norm(target_joints[0, i] - reached_joints[i])
        
        error = reached_pose[0] - desired_pos
        pred_joints["ik/x_error"] = torch.norm( error[0])
        pred_joints["ik/y_error"] = torch.norm( error[1])
        pred_joints["ik/z_error"] = torch.norm( error[2])

        # get action error
        wandb.log(pred_joints)

        # print("target pose", desired_pos, desired_quat, "reached", reached_pose)
        # print("target joint", target_joints, "reached", reached_joints)
        # self.traj_helper.execute_position_path(target_joints)

    def move_to_ee_pose_background(self, desired_pos, desired_quat ):
        # Interpolate a few frames
        # get joint trajectory (diffiktraj) + execute
        # desired_quat = torch.tensor(desired_quat)
        # pose_mat_des = poly_util.polypose2mat((desired_pos, torch.tensor(desired_quat)))

        # ee_poses_interpolated = self.planning.get_diffik_traj(pose_mat_des,from_current=True, N=10)
        # print("TODO: simplify the joint diffik and simply do what we do in sim, even use the same class")
        
        # target_joint_orig = self.traj_helper.diffik_traj(ee_poses_interpolated, total_time=0.5, precompute=True, execute=False )
        
        current_pos, current_rot = self._robot.get_ee_pose()
        # current_ee_pose_mat = polypose2mat(current_ee_pose)

        current_joint_pos = self._robot.get_joint_positions()

        jacobian = self.traj_helper.robot_model.compute_jacobian(current_joint_pos)
        
        target_joint_delta = diffik_step(current_pos.to("cuda"), current_rot.to("cuda"), desired_pos.to("cuda"), desired_quat.to("cuda"), jacobian.to("cuda"))

        target_joints = current_joint_pos.to("cuda") + target_joint_delta
        # TODO: execute the trajectory with the function from traj_helper_diffik_traj

        # interp_steps = 20
        # total_loop_time = 0.5
        # forward_pass_time = 0.13
        total_loop_time = self.total_loop_time
        forward_pass_time = self.forward_pass_time
        with self.target_lock:
            self.new_target = True
            self.target_joints = target_joints
            self.current_pos = self._robot.get_joint_positions() #current_joint_pos
            # interp_joint_pos = np.linspace(self.current_pos.cpu(), self.target_joints.squeeze().cpu(), self.interp_steps).tolist()[1:]
            # self.interp_joint_pos = interp_joint_pos
        if total_loop_time - forward_pass_time > 0:
            time.sleep(total_loop_time - forward_pass_time)

        reached_pose = self._robot.get_ee_pose()
        reached_joints = self._robot.get_joint_positions()

        pred_joints = {}
        for i in range(len(reached_joints)):
            pred_joints['ik/'+str(i)+"_error"] = torch.norm(target_joints[0, i] - reached_joints[i])
        
        error = reached_pose[0] - desired_pos
        pred_joints["ik/x_error"] = torch.norm( error[0])
        pred_joints["ik/y_error"] = torch.norm( error[1])
        pred_joints["ik/z_error"] = torch.norm( error[2])

        # get action error
        wandb.log(pred_joints)

        reached_pose = self._robot.get_ee_pose()
        reached_joints = self._robot.get_joint_positions()
        print("target pose", desired_pos, desired_quat, "reached", reached_pose)
        print("target joint", target_joints, "reached", reached_joints)
        # self.traj_helper.execute_position_path(target_joints)
    
    def joint_control_thread(self):

        interp_joint_pos = []
        # delay = 1.0 / self.interp_steps
        delay = self.total_loop_time / (self.interp_steps - self.start_interp_offset)
        lock_t = 0.0
        idx = 0
        while True:
            with self.target_lock:
                if self.new_target:
                    # interp from current to new target
                    self.current_pos = self._robot.get_joint_positions()
                    interp_joint_pos = np.linspace(self.current_pos.cpu(), self.target_joints.squeeze().cpu(), self.interp_steps).tolist()[self.start_interp_offset:]                 
                    self.new_target = False
                    idx = 0
            
            # if we have any values in the interpolated joints array, pop one and send command
            start = time.time()
            if len(interp_joint_pos) > idx:
                jnt = torch.Tensor(interp_joint_pos[idx]).float()
                try:
                    self._robot.update_desired_joint_positions(jnt)
                except:
                    print("Error here!!!!!!!!!!!!!!!!!!!!!!")

                    self._robot.start_joint_impedance(Kx=self.Kx, Kxd=self.Kxd)
                    print("restarted joint impedance")
                idx += 1

            time.sleep(delay - lock_t)

    def gripper_open(self,):
        print("gripper_open", self._gripper.get_state().width)
        return self._gripper.get_state().width > 0.07
    
    def reset_gripper(self):
        self._gripper.goto(0.08, 0.05, 0.01)
    
    def init_gripper(self):
        self._robot.update_gripper(-1)
        
    def seed(self, seed):
        np.random.seed(seed)
        if self.sim:
            self._robot.seed(seed)
        
    def reset(self):
        self.reset_gripper()
        self._robot.move_to_joint_positions(np.array(self.reset_joints))
        self._robot.start_joint_impedance(Kx=self.Kx, Kxd=self.Kxd)
        
        self._curr_path_length = 0
        self._episode_count += 1
        obs = self.get_observation()

        return obs
    
    @property
    def _curr_pos(self):
        return self._robot.get_ee_pose()[0]

    @property
    def _curr_angle(self):
        return self._robot.get_ee_pose()[1]

    def get_pcd(self):
        return self.camera.get_pcd()

    def get_rgb(self):
        return self.camera.get_rgb()

    def convert_wrist2tip(self, wrist_pose_list):
        """
        Function to convert a pose of the wrist link (nominally, panda_link8) to
        the pose of the frame in between the panda hand fingers
        """        
        # wrist2tip_tf = [0.0, 0.0, 0.1034, 0.0, 0.0, -0.3826834323650898, 0.9238795325112867]
        wrist2tip_tf = [0.0, 0.0, 0.1034, 0.0, 0.0, 0.0, 1.0]  # no rotation (pure z offset)

        tip_pose = util.convert_reference_frame(
            pose_source=util.list2pose_stamped(wrist2tip_tf),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(wrist_pose_list)
        )
        return util.pose_stamped2list(tip_pose)

    def convert_tip2wrist(self, tip_pose_list):
        
        """
        Function to convert a pose of the wrist link (nominally, panda_link8) to
        the pose of the frame in between the panda hand fingers
        """
        # tip2wrist_tf = [0.0, 0.0, -0.1034, 0.0, 0.0, 0.3826834323650898, 0.9238795325112867]
        tip2wrist_tf = [0.0, 0.0, -0.1034, 0.0, 0.0, 0.0, 1.0]

        tip_pose = util.convert_reference_frame(
            pose_source=util.list2pose_stamped(tip2wrist_tf),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(tip_pose_list)
        )
        return util.pose_stamped2list(tip_pose)        
    
    def get_state(self):
        state_dict = {}
        gripper_state = self.get_gripper_state()

        state_dict['control_key'] = 'current_pose'

        # state_dict['current_pose'] = np.concatenate(
        #     [
        #         gripper_state,
        #         self._robot.get_ee_pose()[1],
        #         self._robot.get_ee_pose()[0],
        #     ])

        wrist_pose_tuple = self._robot.get_ee_pose()
        wrist_pose_list = list(wrist_pose_tuple[0]) + list(wrist_pose_tuple[1])
        tip_pose_list = self.convert_wrist2tip(wrist_pose_list) 

        state_dict['current_pose'] = np.concatenate(
            [
                gripper_state,
                tip_pose_list[3:],
                tip_pose_list[:3],
            ])

        state_dict['joint_positions'] = self._robot.get_joint_positions()
        state_dict['joint_velocities'] = self._robot.get_joint_velocities()
        # don't track gripper velocity
        state_dict['gripper_velocity'] = 0

        return state_dict

    def get_gripper_state(self):
        if self.gripper_open():
            gripper_joints = 0.99
        else:
            gripper_joints = -0.98

        return torch.tensor([gripper_joints, gripper_joints])
    
    def get_robot_joints(self):
        arm_joints = self._robot.get_joint_positions()
        if self.gripper_open():
            gripper_joints = 0.99
        else:
            gripper_joints = -0.98

        joints = torch.hstack([arm_joints, torch.tensor([gripper_joints, gripper_joints])]).reshape(1,-1)
        return joints

    def get_observation(self):
        # get state and images
        current_state = self.get_state()
        ee_pos = current_state['current_pose'][-3:]
        gripper_state = self.get_gripper_state()
        ee_pose = self._robot.get_ee_pose()

        
        gripper_state = (self._gripper.get_state().width / 2 )/ 0.04
        ee_quat = ee_pose[1]
        position = ee_pose[0]

        updated_quat = R.from_quat(ee_quat).as_matrix()@R.from_euler('z', -45, degrees=True).as_matrix()
        updated_quat = R.from_matrix(updated_quat).as_quat()

        new_quat = [updated_quat[-1], updated_quat[0], updated_quat[1],updated_quat[2]]
        # new_pos = ee.numpy()#[::-1]
        if self.gripper_open():
            new_gripper_state = [0.99, 0.99]
        else:
            new_gripper_state = [-1, -1]
            # new_gripper_state = [-0.76, -0.75]

        state = np.concatenate([new_gripper_state, new_quat, ee_pos])
        # ee_quat = ee_pose[2:6]
        # new_quat = np.concatenate([[ee_quat[-1]], ee_quat[:-1]])
        # ee_pose[2:6] = new_quat
        # ee_pos = ee_pose[-3:]
        # mid = ee_pos[0]
        # ee_pos[0] = ee_pos[-1]
        # ee_pos[-1] =  mid
        # ee_pose[-3:] = ee_pos

        rgb = self.camera.get_rgb()
        pcd = self.camera.get_pcd()
        joints = self.get_robot_joints()

        obs_dict = {
            'ee_pose': state,
            'joint': joints,
            'rgb': rgb,
            'joints': joints
        }
        return obs_dict

    def render(self, mode=None):
        return self.camera.get_images()
    