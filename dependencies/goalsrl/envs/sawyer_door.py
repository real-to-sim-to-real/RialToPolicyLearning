from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy

from multiworld.core.multitask_env import MultitaskEnv
import matplotlib.pyplot as plt
import os.path as osp
from goalsrl.envs.gymenv_wrapper import GymGoalEnvWrapper
from goalsrl.envs.env_utils import DiscretizedActionEnv, ImageEnv
from multiworld.envs.mujoco.sawyer_xyz import sawyer_door_hook

door_configs = {
    'all': dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
    
}

class SawyerViews:
    @staticmethod
    def configure_viewer(cam, cam_pos):
        for i in range(3):
            cam.lookat[i] = cam_pos[i]
        cam.distance = cam_pos[3]
        cam.elevation = cam_pos[4]
        cam.azimuth = cam_pos[5]
        cam.trackbodyid = -1
    
    @staticmethod
    def robot_view(cam):
        rotation_angle = 90
        cam_dist = 1
        cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def third_person_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def top_down_view(cam):
        cam_dist = 0.2
        rotation_angle = 0
        cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)
    
    @staticmethod
    def default_view(cam):
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 0.85, 0.30, cam_dist, -55, rotation_angle])
        SawyerViews.configure_viewer(cam, cam_pos)

class SawyerDoorGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=False, fixed_goal=False, images=False, image_kwargs=None):
        config_key = 'all'
        if fixed_start:
            if fixed_goal:
                config_key = 'fixed_start_fixed_goal'
            else:
                config_key = 'all'#'fixed_start'
        env = sawyer_door_hook.SawyerDoorHookEnv(**door_configs[config_key])
        
        if images:
            config = dict(init_camera=SawyerViews.default_view, imsize=84, normalize=True, channels_first=True, )
            if image_kwargs is not None:
                config.update(image_kwargs)
            env = ImageEnv(env, **config)

        super(SawyerDoorGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
    
    def extract_goal(self, states):
        original_goal = super().extract_goal(states)
        #original_goal[:3] = 0
        return original_goal
        #return original_goal[]

    def endeff_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 0:3], axis=-1)
    
    def goal_distance(self, states, goal_states):
        return self.door_distance(states, goal_states) + self.endeff_distance(states, goal_states)
    
    def door_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.linalg.norm(diff[..., 3:4], axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        endeff_distances = np.array([self.endeff_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        puck_distances = np.array([self.door_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])

        endeff_movement = self.endeff_distance(trajectories[:,0], trajectories[:, -1])
        puck_movement = self.door_distance(trajectories[:,0], trajectories[:, -1])
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final door distance', puck_distances[:,-1]),
            ('final endeff distance', endeff_distances[:,-1]),
            ('min door distance', np.min(puck_distances, axis=-1)),
            ('min endeff distance', np.min(endeff_distances,axis=-1)),
            ('min distance', np.min(endeff_distances + puck_distances,axis=-1)),
            ('door movement', puck_movement),
            ('endeff movement', endeff_movement),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics
        

def main():
    from rlutil.logging import logger
    e = SawyerDoorGoalEnv(discrete_action=True, fixed_start=True)
    for traj in range(20):
        desired_goal_state = e.sample_goal()
        states = []
        s = e.reset()
        for step in range(1):
            states.append(s)
            s, _, _, _ = e.step(e.action_space.sample())
            #e.render()
        states = np.stack(states)

if __name__ == "__main__":
    main()
