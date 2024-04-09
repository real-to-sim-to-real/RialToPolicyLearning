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

"""Standing tasks with D'Kitty robots.

The goal is to stand upright from an initial configuration.
"""

import abc
import collections
from typing import Dict, Optional, Sequence

import numpy as np

from robel.components.tracking import TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_stand-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'pose_error',
)


class BaseDKittyStand(BaseDKittyUprightEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty turn tasks."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            frame_skip: int = 40,
            upright_threshold: float = 0,  # cos(90deg)
            upright_reward: float = 2,
            falling_reward: float = -100,
            **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            device_path: The device path to Dynamixel hardware.
            torso_tracker_id: The device index or serial of the tracking device
                for the D'Kitty torso.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._desired_pose = np.zeros(12)
        self._initial_pose = np.zeros(12)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing(kitty_pos=self._initial_pose,)
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
        })

        # Let gravity pull the simulated robot to the ground before starting.
        if not self.robot.is_hardware:
            self.robot.step({'dkitty': self._initial_pose}, denormalize=False)
            self.sim_scene.advance(100)

    def _step(self, action: np.ndarray):
        """Applies an action to the robot."""
        self.robot.step({
            'dkitty': action,
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        torso_track_state = self.tracker.get_state('torso')

        return collections.OrderedDict((
            # Add observation terms relating to being upright.
            *self._get_upright_obs(torso_track_state).items(),
            ('root_pos', torso_track_state.pos),
            ('root_euler', torso_track_state.rot_euler),
            ('root_vel', torso_track_state.vel),
            ('root_angular_vel', torso_track_state.angular_vel),
            ('kitty_qpos', robot_state.qpos),
            ('kitty_qvel', robot_state.qvel),
            ('last_action', self._get_last_action()),
            ('pose_error', self._desired_pose - robot_state.qpos),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        pose_mean_error = np.abs(obs_dict['pose_error']).mean(axis=1)
        upright = obs_dict[self._upright_obs_key]
        center_dist = np.linalg.norm(obs_dict['root_pos'][:2], axis=1)

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for closeness to desired pose.
            ('pose_error_cost', -4 * pose_mean_error),
            # Reward for closeness to center; i.e. being stationary.
            ('center_distance_cost', -2 * center_dist),
            # Bonus when mean error < 30deg, scaled by uprightedness.
            ('bonus_small', 5 * (pose_mean_error < (np.pi / 6)) * upright),
            # Bonus when mean error < 15deg and upright within 30deg.
            ('bonus_big',
             10 * (pose_mean_error < (np.pi / 12)) * (upright > 0.9)),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        # Normalize pose error by 60deg.
        pose_points = (1 - np.maximum(
            np.abs(obs_dict['pose_error']).mean(axis=1) / (np.pi / 3), 1))

        return collections.OrderedDict((
            ('points', pose_points * obs_dict['upright']),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))


@configurable(pickleable=True)
class DKittyStandFixed(BaseDKittyStand):
    """Stand up from a fixed position."""

    def _reset(self):
        """Resets the environment."""
        self._initial_pose[[0, 3, 6, 9]] = 0
        self._initial_pose[[1, 4, 7, 10]] = np.pi / 4
        self._initial_pose[[2, 5, 8, 11]] = -np.pi / 2
        super()._reset()


@configurable(pickleable=True)
class DKittyStandRandom(BaseDKittyStand):
    """Stand up from a random position."""

    def _reset(self):
        """Resets the environment."""
        limits = self.robot.get_config('dkitty').qpos_range
        self._initial_pose = self.np_random.uniform(
            low=limits[:, 0], high=limits[:, 1])
        super()._reset()


@configurable(pickleable=True)
class DKittyStandRandomDynamics(DKittyStandRandom):
    """Stand up from a random positon with randomized dynamics."""

    def __init__(self,
                 *args,
                 sim_observation_noise: Optional[float] = 0.05,
                 **kwargs):
        super().__init__(
            *args, sim_observation_noise=sim_observation_noise, **kwargs)
        self._randomizer = SimRandomizer(self)
        self._dof_indices = (
            self.robot.get_config('dkitty').qvel_indices.tolist())

    def _reset(self):
        """Resets the environment."""
        # Randomize joint dynamics.
        self._randomizer.randomize_dofs(
            self._dof_indices,
            all_same=True,
            damping_range=(0.1, 0.2),
            friction_loss_range=(0.001, 0.005),
        )
        self._randomizer.randomize_actuators(
            all_same=True,
            kp_range=(2, 4),
        )
        # Randomize friction on all geoms in the scene.
        self._randomizer.randomize_geoms(
            all_same=True,
            friction_slide_range=(0.8, 1.2),
            friction_spin_range=(0.003, 0.007),
            friction_roll_range=(0.00005, 0.00015),
        )
        # Generate a random height field.
        self._randomizer.randomize_global(
            total_mass_range=(1.6, 2.0),
            height_field_range=(0, 0.05),
        )
        self.sim_scene.upload_height_field(0)
        super()._reset()
