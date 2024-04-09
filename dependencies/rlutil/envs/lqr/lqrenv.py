import itertools

import gym
from gym.spaces import Box
from gym.spaces import Discrete

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rllab.misc import logger

from rlutil.envs.env_utils import flat_to_one_hot
import rlutil.log_utils as log_utils
from rlutil.math_utils import np_seed
from rlutil.envs.lqr.lqr_solver import solve_lqr_env

class LQREnv(gym.Env):
    def __init__(self, 
            initial_state, dA, dynamics_matrix, 
            rew_Q, rew_R, rew_q, 
            rew_bias=0,
            frameskip=10,
            return_values=False,
            return_values_discount=0.99,
            ):
        """
        The reward is given by:
        x^T*req_Q*x + u^T*rew_R*u + x^T*rew_q + rew_bias
        """
        super(LQREnv, self).__init__()
        self.x0 = initial_state
        self.dO = self.x0.shape[0]
        self.dA = dA

        self.dynamics = dynamics_matrix
        self.rew_Q = rew_Q
        self.rew_q = rew_q
        self.rew_R = rew_R
        self.rew_bias = rew_bias
        self.frameskip = frameskip

        self.__observation_space = Box(-1.0, 1.0, shape=(self.dO,))
        self.__action_space = Box(-1.0, 1.0, shape=(self.dA,))
        self.__x = None

        if return_values:
            self.compute_values(discount=return_values_discount)
        self.return_values = return_values

    def _wrap_obs(self, x):
        return x

    def eval_reward(self, x, u):
        return x.T.dot(self.rew_Q).dot(x) + self.rew_q.dot(x) + u.T.dot(self.rew_R).dot(u) + self.rew_bias

    def reset(self):
        self.__x = self.x0
        return self._wrap_obs(self.__x)

    def step(self, u):
        r = self.eval_reward(self.__x, u)
        x = self.__x
        old_x = x
        for _ in range(self.frameskip):
            x = self.dynamics.dot(np.r_[x, u])
        self.__x = x
        done = False

        infos = {'true_state': self.__x}
        if self.return_values:
            infos['value'] = self.value_at(old_x)
            infos['qvalue'] = self.qvalue_at(old_x, u)
        return self._wrap_obs(self.__x), r, done, infos

    def compute_values(self, discount=0.99, K=500):
        self._K, self._k, self._V, self._v, self._Q, self._q = solve_lqr_env(self, discount=discount, solve_itrs=K)
        self._values_computed = True

    def value_at(self, state):
        assert hasattr(self, '_values_computed'), "Please call env.compute_values()"
        return state.T.dot(self._V).dot(state) + self._v.dot(state)

    def qvalue_at(self, state, action):
        assert hasattr(self, '_values_computed'), "Please call env.compute_values()"
        sa = np.r_[state, action]
        return sa.T.dot(self._Q).dot(sa) + self._q.dot(sa)

    def log_diagnostics(self, paths):
        pass

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space


class PointmassEnvVelocity(LQREnv):
    def __init__(self, initial_pos=None, goal_pos=None, action_penalty=1e-2, dt=0.1, sim_steps=1, **kwargs):
        if initial_pos is None:
            initial_pos = np.zeros(2)
        if goal_pos is None:
            goal_pos = np.zeros(2)
        self.goal_pos = goal_pos

        initial_state = initial_pos
        rew_R = - np.eye(2) * action_penalty
        rew_Q = - np.eye(2)
        rew_q = 2*goal_pos
        rew_bias = -np.inner(goal_pos, goal_pos)

        d_sim = dt/sim_steps 
        dynamics_matrix = np.array([
            [1.0, 0.0, d_sim, 0.0],
            [0.0, 1.0, 0.0, d_sim]])
        super(PointmassEnvVelocity, self).__init__(
            initial_state, 2, dynamics_matrix,
            rew_Q, rew_R, rew_q,
            rew_bias=rew_bias,
            frameskip=sim_steps,
            **kwargs
        )


class PointmassEnvTorque(LQREnv):
    def __init__(self, initial_pos=None, goal_pos=None, 
            action_penalty=1e-2, dt=0.1, sim_steps=1, gains=10, **kwargs):
        if initial_pos is None:
            initial_pos = np.zeros(2)
        if goal_pos is None:
            goal_pos = np.zeros(2)
        self.goal_pos = goal_pos
        initial_state = np.r_[initial_pos, np.zeros(2)]
        rew_R = - np.eye(2) * action_penalty
        rew_Q = - np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            ])
        rew_q = np.r_[2*goal_pos, np.zeros(2)]
        rew_bias = -np.inner(goal_pos, goal_pos)

        d_sim = dt/sim_steps 
        dynamics_matrix = np.array([
            [1.0, 0.0, d_sim, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, d_sim, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, gains*0.5*d_sim**2, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, gains*0.5*d_sim**2],
            ])
        super(PointmassEnvTorque, self).__init__(
            initial_state, 2, dynamics_matrix,
            rew_Q, rew_R, rew_q,
            rew_bias=rew_bias,
            frameskip=sim_steps,
            **kwargs
        )


class PointmassEnvVision(PointmassEnvTorque):
    def __init__(self, im_width=64, im_height=64, **kwargs):
        self.w = im_width
        self.h = im_height
        super(PointmassEnvVision, self).__init__(**kwargs)
        self.__observation_space = Box(0.0, 1.0, shape=(self.w, self.h))

    def _wrap_obs(self, x):
        bx, by = [-2, 2], [-2, 2]

        image = np.zeros((self.w, self.h))
        x, y = x[0], x[1]
        gx, gy = self.goal_pos

        pix_x = int((x-bx[0])/(bx[1]-bx[0]) * self.w)
        pix_y = int((y-by[0])/(by[1]-by[0]) * self.h)
        pix_gx = int((gx-bx[0])/(bx[1]-bx[0]) * self.w)
        pix_gy = int((gy-by[0])/(by[1]-by[0]) * self.h)

        image[pix_x, pix_y] = 1.0
        image[pix_gx, pix_gy] = 1.0
        return image

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = PointmassEnvVision(return_values=True)
    obs = env.reset()

    for _ in range(100):
        obs, _, _, infos = env.step(10*env.action_space.sample())
        print(infos)
        plt.imshow(obs)
        plt.show()
        import pdb; pdb.set_trace()

