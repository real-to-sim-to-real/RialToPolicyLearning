"""
Implements useful utility functions:

- Discretized (gym.space): A gym space corresponding to a gym.spaces.Box space
    which was discretized per dimension. Extends gym.spaces.Discrete to expose
    the number of dimensions and the granularity of the discretization
- DiscretizedActionEnv: Wraps a continuous action environment into a
    discrete action environment by discretizing per-dimension
- ImageEnv: Wraps a Multiworld env to change observation space into images
    (copied primarily from multiworld)
"""
from textwrap import wrap
import numpy as np
import warnings

import gym
from gym.spaces import Dict, Box, Discrete

from multiworld.core.wrapper_env import ProxyEnv

from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

import copy 

# Images
from PIL import Image
class ImageandProprio(Box):
    def __init__(self, image_shape, proprio_shape):
        self.image_shape = image_shape
        self.proprio_shape = proprio_shape
        example = self.to_flat(np.zeros(self.image_shape), np.zeros(self.proprio_shape))
        super(ImageandProprio, self).__init__(0, 1, shape=example.shape)
    def to_flat(self, image, proprio):
        image = image.reshape(*image.shape[:-1 * len(self.image_shape)], -1)
        proprio = proprio.reshape(*proprio.shape[:-1 * len(self.proprio_shape)], -1)
        return np.concatenate([image, proprio], axis=-1)

    def from_flat(self, s):
        image_size = np.prod(self.image_shape)
        image = s[..., :image_size]
        image = image.reshape(*image.shape[:-1], *self.image_shape)
        proprio = s[..., image_size:]
        proprio = proprio.reshape(*proprio.shape[:-1], *self.proprio_shape)
        return image, proprio


class Discretized(Discrete):
    def __init__(self, n, n_dims, granularity):
        self.n_dims = n_dims
        self.granularity = granularity
        assert n == granularity ** n_dims
        super(Discretized, self).__init__(n)

class DummyWrappedEnv(ProxyEnv):
    def __init__(self, wrapped_env, possible_actions=None, granularity=3):
        self.quick_init(locals())
        ProxyEnv.__init__(self, wrapped_env)

    def step(self, action):
        return self.wrapped_env.step(action)

class DiscretizedActionEnv(ProxyEnv):
    def __init__(self, wrapped_env, possible_actions=None, granularity=3):
        self.quick_init(locals())
        ProxyEnv.__init__(self, wrapped_env)
        if possible_actions is not None:
            self.base_actions = possible_actions
            n_dims = 1
            granularity = len(self.base_actions)
        
        else:
            actions_meshed = np.meshgrid(*[np.linspace(lo, hi, granularity) for lo, hi in zip(self.wrapped_env.action_space.low, self.wrapped_env.action_space.high)])
            self.base_actions = np.array([a.flat[:] for a in actions_meshed]).T
            n_dims = self.wrapped_env.action_space.shape[0]
        self.action_space = Discretized(len(self.base_actions), n_dims, granularity)

    def step(self, action):
        return self.wrapped_env.step(self.base_actions[action])

class DiscretizedActionRavensEnv(ProxyEnv):
    def __init__(self, wrapped_env, possible_actions=None, actions_to_discretize = ["pose0", "pose1"], granularity=3):
        self.quick_init(locals())
        ProxyEnv.__init__(self, wrapped_env)

        self.actions_to_discretize = actions_to_discretize
        self.base_actions = {}

        action_space_dict = {}

        print(self.wrapped_env.base_env._env.action_space)
        for a in actions_to_discretize:
            for idx in range(2):
                a_prime = a +f"_{idx}"
                position_bounds = self.wrapped_env.base_env._env.action_space[a][idx]
                actions_position_meshed = np.meshgrid(*[np.linspace(lo, hi, granularity) for lo, hi in zip(position_bounds.low, position_bounds.high)])
                base_positon_actions = np.array([a.flat[:] for a in actions_position_meshed]).T
                n_dims_position = position_bounds.shape[0]
                
                action_space_dict[a_prime] = Discretized(len(base_positon_actions), n_dims_position, granularity)

                self.base_actions[a_prime] = base_positon_actions
 

        self.action_space = gym.spaces.Dict(action_space_dict)
        

    def step(self, action):
        for a in self.actions_to_discretize:
            a_prime0 = a +f"_{0}"
            a_prime1 = a +f"_{1}"
            action_tuple = self.base_actions[a_prime0], self.base_actions[a_prime1]
            action[a] = action_tuple
        return self.wrapped_env.step(action)
