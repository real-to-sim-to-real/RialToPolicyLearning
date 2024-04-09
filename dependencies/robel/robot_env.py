# Copyright 2019 The ROBEL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base environment API for robotics tasks."""

import abc
import collections
from typing import Any, Dict, Optional, Sequence, Union, Tuple

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from robel.components.builder import ComponentBuilder
from robel.simulation.sim_scene import SimScene, SimBackend
from robel.simulation.renderer import RenderMode

DEFAULT_RENDER_SIZE = 480

# The simulation backend to use by default.
DEFAULT_SIM_BACKEND = SimBackend.MUJOCO_PY


def make_box_space(low: Union[float, Sequence[float]],
                   high: Union[float, Sequence[float]],
                   shape: Optional[Tuple[int]] = None) -> gym.spaces.Box:
    """Returns a Box gym space."""
    # HACK: Fallback for gym 0.9.x
    # TODO(michaelahn): Consider whether we still need to support 0.9.x
    try:
        return spaces.Box(low, high, shape, dtype=np.float32)
    except TypeError:
        return spaces.Box(low, high, shape)


class RobotEnv(gym.Env, metaclass=abc.ABCMeta):
    """Base Gym environment for robotics tasks."""

    def __init__(self,
                 sim_model: Any,
                 observation_keys: Optional[Sequence[str]] = None,
                 reward_keys: Optional[Sequence[str]] = None,
                 use_dict_obs: bool = False,
                 frame_skip: int = 1,
                 camera_settings: Optional[Dict] = None,
                 sim_backend: SimBackend = DEFAULT_SIM_BACKEND,
                 sticky_action_probability: float = 0.):
        """Initializes a robotics environment.

        Args:
            sim_model: The path to the simulation to load.
            observation_keys: The keys of `get_obs_dict` to extract and flatten
                for the default implementation of `_get_obs`. If this is not
                set, `get_obs_dict` must return an OrderedDict.
            reward_keys: The keys of `get_reward_dict` to extract and sum for
                the default implementation of `_get_total_reward`. If this is
                not set, `_get_total_reward` will sum all of the values.
            use_dict_obs: If True, the observations will be returned as
                dictionaries rather than as a flattened array. The observation
                space of this environment will be a dictionary space.
            frame_skip: The number of simulation steps per environment step.
                This multiplied by the timestep defined in the model file is the
                step duration.
            camera_settings: Settings to apply to the free camera in simulation.
            sim_backend: The simulation backend to use.
            sticky_action_probability: Repeat previous action with this
                probability. Default is 0 (no sticky actions).
        """
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._use_dict_obs = use_dict_obs
        self._sticky_action_probability = sticky_action_probability
        self._components = []

        # The following spaces are initialized by their respective `initialize`
        # methods, e.g. `_initialize_observation_space`.
        self._observation_space = None
        self._action_space = None
        self._state_space = None

        # The following are populated by step() and/or reset().
        self.last_action = None
        self.last_obs_dict = None
        self.last_reward_dict = None
        self.last_score_dict = None
        self.is_done = False
        self.step_count = 0

        # Load the simulation.
        self.sim_scene = SimScene.create(
            sim_model, backend=sim_backend, frame_skip=frame_skip)
        self.sim = self.sim_scene.sim
        self.model = self.sim_scene.model
        self.data = self.sim_scene.data

        if camera_settings:
            self.sim_scene.renderer.set_free_camera_settings(**camera_settings)

        # Set common metadata for Gym environments.
        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(
                np.round(1.0 / self.sim_scene.step_duration))
        }
        # Ensure gym does not try to patch `_step` and `_reset`.
        self._gym_disable_underscore_compat = True

        self.seed()

    #===========================================================================
    # Environment API.
    # These methods should not be overridden by subclasses.
    #===========================================================================

    @property
    def observation_space(self) -> gym.Space:
        """Returns the observation space of the environment.

        The observation space is the return specification for `reset`,
        `_get_obs`, and the first element of the returned tuple from `step`.

        Subclasses should override `_initialize_observation_space` to customize
        the observation space.
        """
        # Initialize and cache the observation space on the first call.
        if self._observation_space is None:
            self._observation_space = self._initialize_observation_space()
            assert self._observation_space is not None
        return self._observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the action space of the environment.

        The action space is the argument specifiction for `step`.

        Subclasses should override `_initialize_action_space` to customize the
        action space.
        """
        # Initialize and cache the action space on the first call.
        if self._action_space is None:
            self._action_space = self._initialize_action_space()
            assert self._action_space is not None
        return self._action_space

    @property
    def state_space(self) -> gym.Space:
        """Returns the state space of the environment.

        The state space is the return specification for `get_state` and is the
        argument specification for `set_state`.

        Subclasses should override `_initialize_state_space` to customize the
        state space.
        """
        # Initialize and cache the state space on the first call.
        if self._state_space is None:
            self._state_space = self._initialize_state_space()
            assert self._state_space is not None
        return self._state_space

    @property
    def dt(self) -> float:
        """Returns the step duration of each step, in seconds."""
        return self.sim_scene.step_duration

    @property
    def obs_dim(self) -> int:
        """Returns the size of the observation space.

        NOTE: This is for compatibility with gym.MujocoEnv.
        """
        if not isinstance(self.observation_space, spaces.Box):
            raise NotImplementedError('`obs_dim` only supports Box spaces.')
        return np.prod(self.observation_space.shape).item()

    @property
    def action_dim(self) -> int:
        """Returns the size of the action space."""
        if not isinstance(self.action_space, spaces.Box):
            raise NotImplementedError('`action_dim` only supports Box spaces.')
        return np.prod(self.action_space.shape).item()

    def seed(self, seed: Optional[int] = None) -> Sequence[int]:
        """Seeds the environment.

        Args:
            seed: The value to seed the random number generator with. If None,
                uses a random seed.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> Any:
        """Resets the environment.

        Args:
            state: The state to reset to. This must match with the state space
                of the environment.

        Returns:
            The initial observation of the environment after resetting.
        """
        self.last_action = None
        self.sim.reset()
        self.sim.forward()
        self._reset()

        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        self.last_reward_dict = None
        self.last_score_dict = None
        self.is_done = False
        self.step_count = 0

        return self._get_obs(obs_dict)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Runs one timestep of the environment with the given action.

        Subclasses must override 4 subcomponents of step:
        - `_step`: Applies an action to the robot
        - `get_obs_dict`: Returns the current observation of the robot.
        - `get_reward_dict`: Calculates the reward for the step.
        - `get_done`: Returns whether the episode should terminate.

        Args:
            action: An action to control the environment.

        Returns:
            observation: The observation of the environment after the timestep.
            reward: The amount of reward obtained during the timestep.
            done: Whether the episode has ended. `env.reset()` should be called
                if this is True.
            info: Auxiliary information about the timestep.
        """
        # Perform the step.
        action = self._preprocess_action(action)
        self._step(action)
        self.last_action = action

        # Get the observation after the step.
        obs_dict = self.get_obs_dict()
        self.last_obs_dict = obs_dict
        flattened_obs = self._get_obs(obs_dict)

        # Get the rewards for the observation.
        batched_action = np.expand_dims(np.atleast_1d(action), axis=0)
        batched_obs_dict = {
            k: np.expand_dims(np.atleast_1d(v), axis=0)
            for k, v in obs_dict.items()
        }
        batched_reward_dict = self.get_reward_dict(batched_action,
                                                   batched_obs_dict)

        # Calculate the total reward.
        reward_dict = {k: v.item() for k, v in batched_reward_dict.items()}
        self.last_reward_dict = reward_dict
        reward = self._get_total_reward(reward_dict)

        # Calculate the score.
        batched_score_dict = self.get_score_dict(batched_obs_dict,
                                                 batched_reward_dict)
        score_dict = {k: v.item() for k, v in batched_score_dict.items()}
        self.last_score_dict = score_dict

        # Get whether the episode should end.
        dones = self.get_done(batched_obs_dict, batched_reward_dict)
        done = dones.item()
        self.is_done = done

        # Combine the dictionaries as the auxiliary information.
        info = collections.OrderedDict()
        info.update(('obs/' + key, val) for key, val in obs_dict.items())
        info.update(('reward/' + key, val) for key, val in reward_dict.items())
        info['reward/total'] = reward
        info.update(('score/' + key, val) for key, val in score_dict.items())

        self.step_count += 1

        return flattened_obs, reward, done, info

    def render(
            self,
            mode: str = 'human',
            width: int = DEFAULT_RENDER_SIZE,
            height: int = DEFAULT_RENDER_SIZE,
            camera_id: int = -1,
    ) -> Optional[np.ndarray]:
        """Renders the environment.

        Args:
            mode: The type of rendering to use.
                - 'human': Renders to a graphical window.
                - 'rgb_array': Returns the RGB image as an np.ndarray.
                - 'depth_array': Returns the depth image as an np.ndarray.
            width: The width of the rendered image. This only affects offscreen
                rendering.
            height: The height of the rendered image. This only affects
                offscreen rendering.
            camera_id: The ID of the camera to use. By default, this is the free
                camera. If specified, only affects offscreen rendering.

        Returns:
            If mode is `rgb_array` or `depth_array`, a Numpy array of the
            rendered pixels. Otherwise, returns None.
        """
        if mode == 'human':
            self.sim_scene.renderer.render_to_window()
        elif mode == 'rgb_array':
            return self.sim_scene.renderer.render_offscreen(
                width, height, mode=RenderMode.RGB, camera_id=camera_id)
        elif mode == 'depth_array':
            return self.sim_scene.renderer.render_offscreen(
                width, height, mode=RenderMode.DEPTH, camera_id=camera_id)
        else:
            raise NotImplementedError(mode)
        return None

    def close(self):
        """Cleans up any resources used by the environment."""
        for component in self._components:
            component.close()
        self._components.clear()
        self.sim_scene.close()

    #===========================================================================
    # Overridable Methods
    #===========================================================================

    @abc.abstractmethod
    def _reset(self):
        """Task-specific reset for the environment."""

    @abc.abstractmethod
    def _step(self, action: np.ndarray):
        """Task-specific step for the environment."""

    @abc.abstractmethod
    def get_obs_dict(self) -> Dict[str, Any]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """

    @abc.abstractmethod
    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation.

        Args:
            action: A batch of actions.
            obs_dict: A dictionary of batched observations. The batch dimension
                matches the batch dimension of the actions.

        Returns:
            A dictionary of reward components. The values should be batched to
            match the given actions and observations.
        """

    @abc.abstractmethod
    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment.

        Args:
            obs_dict: A dictionary of batched observations.
            reward_dict: A dictionary of batched rewards to correspond with the
                observations.

        Returns:
            A dictionary of scores.
        """

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate.

        Args:
            obs_dict: A dictionary of batched observations.
            reward_dict: A dictionary of batched rewards to correspond with the
                observations.

        Returns:
            A boolean to denote if the episode should terminate. This should
            have the same batch dimension as the observations and rewards.
        """
        del obs_dict
        return np.zeros_like(next(iter(reward_dict.values())), dtype=bool)

    def get_state(self) -> Any:
        """Returns the current state of the environment."""
        return (self.data.qpos.copy(), self.data.qvel.copy())

    def set_state(self, state: Any):
        """Sets the state of the environment."""
        qpos, qvel = state
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.sim.forward()

    def _initialize_observation_space(self) -> gym.Space:
        """Returns the observation space to use for this environment.

        The default implementation calls `_get_obs()` and returns a dictionary
        space if the observation is a mapping, or a box space otherwise.
        """
        observation = self._get_obs()
        if isinstance(observation, collections.Mapping):
            assert self._use_dict_obs
            return spaces.Dict({
                key: make_box_space(-np.inf, np.inf, shape=np.shape(value))
                for key, value in observation.items()
            })
        return make_box_space(-np.inf, np.inf, shape=observation.shape)

    def _initialize_action_space(self) -> gym.Space:
        """Returns the action space to use for this environment.

        The default implementation uses the simulation's control actuator
        dimensions as the action space, using normalized actions in [-1, 1].
        """
        return make_box_space(-1.0, 1.0, shape=(self.model.nu,))

    def _initialize_state_space(self) -> gym.Space:
        """Returns the state space to use for this environment.

        The default implementation calls `get_state()` and returns a space
        corresponding to the type of the state object:
        - Mapping: Dict space
        - List/Tuple: Tuple space
        """
        state = self.get_state()
        if isinstance(state, collections.Mapping):
            return spaces.Dict({
                key: make_box_space(-np.inf, np.inf, shape=np.shape(value))
                for key, value in state.items()  # pylint: disable=no-member
            })
        elif isinstance(state, (list, tuple)):
            return spaces.Tuple([
                make_box_space(-np.inf, np.inf, shape=np.shape(value))
                for value in state
            ])
        raise NotImplementedError(
            'Override _initialize_state_space for state: {}'.format(state))

    def _get_last_action(self) -> np.ndarray:
        """Returns the previous action, or zeros if no action has been taken."""
        if self.last_action is None:
            return np.zeros((self.action_dim,), dtype=self.action_space.dtype)
        return self.last_action

    def _preprocess_action(self, action: np.ndarray) -> np.ndarray:
        """Transforms an action before passing it to `_step()`.

        Args:
            action: The action in the environment's action space.

        Returns:
            The transformed action to pass to `_step()`.
        """
        # Clip to the normalized action space.
        action = np.clip(action, -1.0, 1.0)

        # Prevent elements of the action from changing if sticky actions are
        # being used.
        if self._sticky_action_probability > 0 and self.last_action is not None:
            sticky_indices = (
                self.np_random.uniform() < self._sticky_action_probability)
            action = np.where(sticky_indices, self.last_action, action)

        return action

    def _get_obs(self, obs_dict: Optional[Dict[str, np.ndarray]] = None) -> Any:
        """Returns the current observation of the environment.

        This matches the environment's observation space.
        """
        if obs_dict is None:
            obs_dict = self.get_obs_dict()
        if self._use_dict_obs:
            if self._observation_keys:
                obs = collections.OrderedDict(
                    (key, obs_dict[key]) for key in self._observation_keys)
            else:
                obs = obs_dict
        else:
            if self._observation_keys:
                obs_values = (obs_dict[key] for key in self._observation_keys)
            else:
                assert isinstance(obs_dict, collections.OrderedDict), \
                    'Must use OrderedDict if not using `observation_keys`'
                obs_values = obs_dict.values()
            obs = np.concatenate([np.ravel(v) for v in obs_values])
        return obs

    def _get_total_reward(self, reward_dict: Dict[str, np.ndarray]) -> float:
        """Returns the total reward for the given reward dictionary.

        The default implementation extracts the keys from `reward_keys` and sums
        the values.

        Args:
            reward_dict: A dictionary of rewards. The values may have a batch
                dimension.

        Returns:
            The total reward for the dictionary.
        """
        # TODO(michaelahn): Enforce that the reward values are scalar.
        if self._reward_keys:
            reward_values = (reward_dict[key] for key in self._reward_keys)
        else:
            reward_values = reward_dict.values()
        return np.sum(np.fromiter(reward_values, dtype=float))

    def _add_component(self, component_builder: ComponentBuilder,
                       **component_kwargs) -> Any:
        """Creates a new component for this environment instance.

        Args:
            component_builder: The configured ComponentBuilder to build the
                component with.
        """
        # Build the component.
        component = component_builder.build(
            sim_scene=self.sim_scene,
            random_state=self.np_random,
            **component_kwargs)
        self._components.append(component)
        return component
