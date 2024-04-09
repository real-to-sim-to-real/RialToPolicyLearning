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
import numpy as np
import warnings

import gym
from gym.spaces import Dict, Box, Discrete

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv

from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

# Images
from PIL import Image

class Discretized(Discrete):
    def __init__(self, n, n_dims, granularity):
        self.n_dims = n_dims
        self.granularity = granularity
        assert n == granularity ** n_dims
        super(Discretized, self).__init__(n)

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

class ImageEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            transpose=False,
            grayscale=False,
            normalize=False,
            reward_type='wrapped_env',
            threshold=10,
            presampled_goals=None,
            non_presampled_goal_img_is_garbage=False,
            recompute_reward=False,
            channels_first=False,
    ):
        """

        :param wrapped_env:
        :param imsize:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        :param non_presampled_goal_img_is_garbage: Set this option to True if
        you want to allow the code to work without presampled goals,
        but where the underlying env doesn't support set_to_goal. As the name,
        implies this will make it so that the goal image is garbage if you
        don't provide pre-sampled goals. The main use case is if you want to
        use an ImageEnv to pre-sample a bunch of goals.
        :param channels_first: if True, then (# Channels, imsize, imsize) instead of (imsize, imsize, # Channels)
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = True
        self.imsize = imsize
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage


        if grayscale:
            self.image_length = self.imsize * self.imsize
        else:
            self.image_length = 3 * self.imsize * self.imsize
        self.channels = 1 if grayscale else 3

        self.channels_first = channels_first
        if channels_first:
            self.image_shape = (self.channels, self.imsize, self.imsize)
        else:
            self.image_shape = (self.imsize, self.imsize, self.channels)

        # Flattened past image queue
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.initialize_camera(init_camera)
            # viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            # init_camera(viewer.cam)
            # sim.add_render_context(viewer)
        
        img_space = Box(0, 1, self.image_shape, dtype=np.float32)
        flat_image_space = Box(0, 1, (self.image_length, ), dtype=np.float32)
        
        self._img_goal = img_space.sample() #has to be done for presampling
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['observation'] = img_space
        spaces['desired_goal'] = img_space
        spaces['achieved_goal'] = img_space
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space
        
        self.return_image_proprio = False
        if 'proprio_observation' in spaces.keys():
            self.return_image_proprio = True
            spaces['image_proprio_observation'] = concatenate_box_spaces(
                flat_image_space,
                spaces['proprio_observation']
            )
            spaces['image_proprio_desired_goal'] = concatenate_box_spaces(
                flat_image_space,
                spaces['proprio_desired_goal']
            )
            spaces['image_proprio_achieved_goal'] = concatenate_box_spaces(
                flat_image_space,
                spaces['proprio_achieved_goal']
            )

        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[np.random.choice(list(presampled_goals))].shape[0]
        self._last_image = None

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            reward = self.compute_reward(action, new_obs)
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        achieved_goal = obs['image_achieved_goal']
        desired_goal = self._img_goal
        image_dist = np.linalg.norm(achieved_goal-desired_goal)
        image_success = (image_dist<self.threshold).astype(float)-1
        info['image_dist'] = image_dist
        info['image_success'] = image_success

    def reset(self):
        obs = self.wrapped_env.reset()
        if self.num_goals_presampled > 0:
            goal = self.sample_goal()
            self._img_goal = goal['image_desired_goal']
            self.wrapped_env.set_goal(goal)
            for key in goal:
                obs[key] = goal[key]
        elif self.non_presampled_goal_img_is_garbage:
            # This is use mainly for debugging or pre-sampling goals.
            self._img_goal = self._get_img()
        else:
            env_state = self.wrapped_env.get_env_state()
            self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())
            self._img_goal = self._get_img()
            self.wrapped_env.set_env_state(env_state)

        return self._update_obs(obs)

    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        img_obs = self._get_img()
        obs['image_observation'] = img_obs
        obs['image_desired_goal'] = self._img_goal
        obs['image_achieved_goal'] = img_obs
        obs['observation'] = img_obs
        obs['desired_goal'] = self._img_goal
        obs['achieved_goal'] = img_obs

        if self.return_image_proprio:
            obs['image_proprio_observation'] = np.concatenate(
                (obs['image_observation'].flatten(), obs['proprio_observation'])
            )
            obs['image_proprio_desired_goal'] = np.concatenate(
                (obs['image_desired_goal'].flatten(), obs['proprio_desired_goal'])
            )
            obs['image_proprio_achieved_goal'] = np.concatenate(
                (obs['image_achieved_goal'].flatten(), obs['proprio_achieved_goal'])
            )

        return obs

    def _get_img(self):
        image_obs = self._wrapped_env.get_image(
            width=self.imsize,
            height=self.imsize,
        )
        image_obs = image_obs[:,::-1, :] # Image is flipped upside down

        self._last_image = image_obs
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        
        if self.channels_first:
            image_obs = np.moveaxis(image_obs, 2, 0)
        
        return image_obs

    def render(self, mode='wrapped'):
        if mode == 'wrapped':
            self.wrapped_env.render()
        elif mode == 'cv2':
            import cv2

            if self._last_image is None:
                self._last_image = self._wrapped_env.get_image(
                    width=self.imsize,
                    height=self.imsize,
                )
            cv2.imshow('ImageEnv', self._last_image)
            cv2.waitKey(1)
        else:
            raise ValueError("Invalid render mode: {}".format(mode))

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._img_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)

    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")
        img_goals = np.zeros((batch_size, *self.image_shape))
        goals = self.wrapped_env.sample_goals(batch_size)
        pre_state = self.wrapped_env.get_env_state()
        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)
            img_goals[i] = self._get_img()
        self.wrapped_env.set_env_state(pre_state)
        goals['desired_goal'] = img_goals
        goals['image_desired_goal'] = img_goals
        return goals

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type=='image_distance':
            return - dist
        elif self.reward_type=='image_sparse':
            return -(dist > self.threshold).astype(float)
        elif self.reward_type=='wrapped_env':
            return self.wrapped_env.compute_rewards(actions, obs)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

def normalize_image(image, dtype=np.float64):
    assert image.dtype == np.uint8
    return dtype(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
