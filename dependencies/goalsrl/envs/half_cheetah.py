import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from goalsrl.envs import goal_env

class GoalHalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle, goal_env.GoalEnv):
    def __init__(self, action_ratio=1.0, uniform_reset=False):
        self.action_ratio = action_ratio
        mujoco_env.MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        goal_env.GoalEnv.__init__(self)
        utils.EzPickle.__init__(self)

        self.state_space = self.observation_space
        self.uniform_reset = uniform_reset
        #self.observation_space = gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(17,), dtype=np.float32)
        v = np.ones(1)
        self.goal_scale = goal_scale = 3.0
        self.goal_space = gym.spaces.Box(low=-goal_scale*v, high=goal_scale*v, dtype=np.float32)

    def step(self, action):
        self.do_simulation(action * self.action_ratio, self.frame_skip)
        return self.get_state(), 0, False, {}

    def get_state(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=0., high=0., size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .0
        if self.uniform_reset:
            qpos[0] = self.np_random.uniform(low=-self.goal_scale, high=self.goal_scale)
        self.set_state(qpos, qvel)
        return self.get_state()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def observation(self, state):
        return state

    def extract_goal(self, state):
        return state[..., :1]

    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            return np.linalg.norm(state[..., :1] - goal_state[..., :1], axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)
