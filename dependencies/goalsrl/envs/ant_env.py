"""
A GoalEnv with the low-gear ratio Ant

Observation Space (29 dim): QPos + QVel 
Goal Space (2 dim): COM Position
Action Space (8 dim): Joint Torque Control
"""

import numpy as np
import gym
from gym import utils
from gym.envs.mujoco import mujoco_env
import os.path as osp
from goalsrl.envs import goal_env

from collections import OrderedDict
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

class GoalAntEnv(mujoco_env.MujocoEnv, utils.EzPickle, goal_env.GoalEnv):
    def __init__(self, size=4, action_ratio=1, fixed_start=True, fixed_goal=False):
        self.size = size
        self.action_ratio = action_ratio

        self.fixed_start = fixed_start
        self.fixed_goal = fixed_goal
        self.steps = 0

        model_name = osp.abspath(osp.join(osp.dirname(__file__), 'assets/ant.xml'))
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        goal_env.GoalEnv.__init__(self)
        utils.EzPickle.__init__(self, size, action_ratio, fixed_start, fixed_goal)

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32) # 29
        self.state_space =  gym.spaces.Box(low=-1, high=1, shape=(29,), dtype=np.float32)
        self.goal_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) #15

    def step(self, a):
        self.do_simulation(a * self.action_ratio, self.frame_skip)
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        self.steps += 1
        if done and not self.done:
            # print('Done!', self.steps)
            # print(np.isfinite(state).all(), state[2] >= 0.2, state[2] <= 1.0)
            self.done = True
        done = False

        return self.get_state(), 0, done, {}

    def get_state(self):
        return np.concatenate([
            self.sim.data.qpos.flat[:15],
            self.sim.data.qvel.flat[:14],
        ])

    def observation(self, state):
        return state[..., :15]

    def extract_goal(self, state):
        return state[..., :2]
    
    def _extract_sgoal(self, state):
        return state[..., :2]

    def reset(self):
        self.done = False
        self.steps = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        if not self.fixed_start:
            new_position = np.random.rand(2) * self.size - self.size / 2
            qpos[:2] = new_position

        self.set_state(qpos, qvel)
        return self.get_state()

    def sample_goal(self):
        state = self.reset()
        if self.fixed_goal:
            new_position = np.random.ones(2) * self.size / 2
        else:
            new_position = np.random.rand(2) * self.size - self.size / 2

        # Move self to a weird spot
        a = self.action_space.sample()
        for _ in range(5):
            self.do_simulation(a * self.action_ratio, self.frame_skip)
            a = 0.5 * a + 0.5 * self.action_space.sample()
        state = self.get_state()
        state[:2] += new_position
        self.reset()
        return state

    def extract_position(self, state):
        return state[..., :2]

    def goal_distance(self, state, goal_state):
        return np.linalg.norm(self.extract_position(state) - self.extract_position(goal_state), axis=-1)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_image(self, state, imsize=84, channels_first=False):
        old_qpos, old_qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        
        qpos = self.init_qpos.copy()
        qpos[:15] = state[:15]
        qvel = self.init_qvel.copy()
        qvel[:14] = state[15:29]

        self.set_state(qpos, qvel)
        image_obs = self.sim.render(imsize, imsize).copy()
        image_obs = image_obs[:,::-1, :]
        if channels_first:
            image_obs = np.moveaxis(image_obs, 2, 0)
        
        self.set_state(old_qpos, old_qvel)
        return image_obs


    def get_diagnostics(self, trajectories, desired_goal_states):
        minv = min([len(trajectory) for trajectory in trajectories])
        trajectories = np.array([trajectory[:minv] for trajectory in trajectories])

        total_distances = np.array([np.linalg.norm(trajectories[i] - np.tile(desired_goal_states[i], (trajectories.shape[1],1)), axis=-1) for i in range(trajectories.shape[0])])
        com_distances = np.array([self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        com_movement = self.goal_distance(trajectories[:,0], trajectories[:, -1])
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final COM distance', com_distances[:,-1]),
            ('final total distance', total_distances[:,-1]),
            ('COM movement', com_movement),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics