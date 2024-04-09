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

"""Orient tasks with D'Kitty robots.

The goal is to change the orientation of the D'Kitty in place.
"""

import abc
import collections
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
from transforms3d.euler import euler2quat

from robel.components.tracking import TrackerComponentBuilder, TrackerState
from robel.dkitty.base_env import BaseDKittyUprightEnv
from robel.simulation.randomize import SimRandomizer
from robel.utils.configurable import configurable
from robel.utils.math_utils import calculate_cosine
from robel.utils.resources import get_asset_path

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_orient-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'kitty_qvel',
    'last_action',
    'upright',
    'current_facing',
    'desired_facing',
)


class BaseDKittyOrient(BaseDKittyUprightEnv, metaclass=abc.ABCMeta):
    """Shared logic for DKitty orient tasks."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            target_tracker_id: Optional[Union[str, int]] = None,
            frame_skip: int = 40,
            upright_threshold: float = 0.9,  # cos(~25deg)
            upright_reward: float = 2,
            falling_reward: float = -500,
            **kwargs):
        """Initializes the environment.

        Args:
            asset_path: The XML model file to load.
            observation_keys: The keys in `get_obs_dict` to concatenate as the
                observations returned by `step` and `reset`.
            target_tracker_id: The device index or serial of the tracking device
                for the target.
            frame_skip: The number of simulation steps per environment step.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
        """
        self._target_tracker_id = target_tracker_id

        super().__init__(
            sim_model=get_asset_path(asset_path),
            observation_keys=observation_keys,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

        self._initial_angle = 0
        self._target_angle = 0

        self._markers_bid = self.model.body_name2id('markers')
        self._current_angle_bid = self.model.body_name2id('current_angle')
        self._target_angle_bid = self.model.body_name2id('target_angle')

    def _configure_tracker(self, builder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'target',
            hardware_tracker_id=self._target_tracker_id,
            sim_params=dict(
                element_name='target',
                element_type='site',
            ),
            mimic_xy_only=True)

    def _reset(self):
        """Resets the environment."""
        self._reset_dkitty_standing()
        # Set the initial target position.
        self.tracker.set_state({
            'torso': TrackerState(
                pos=np.zeros(3),
                rot_euler=np.array([0, 0, self._initial_angle])),
            'target': TrackerState(
                pos=np.array([
                    # The D'Kitty is offset to face the y-axis.
                    np.cos(self._target_angle + np.pi / 2),
                    np.sin(self._target_angle + np.pi / 2),
                    0,
                ])),
        })

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
        torso_track_state, target_track_state = self.tracker.get_state(
            ['torso', 'target'])

        # Get the facing direction of the kitty. (the y-axis).
        current_facing = torso_track_state.rot[:2, 1]

        # Get the direction to the target.
        target_facing = target_track_state.pos[:2] - torso_track_state.pos[:2]
        target_facing = target_facing / np.linalg.norm(target_facing + 1e-8)

        # Calculate the alignment to the facing direction.
        angle_error = np.arccos(calculate_cosine(current_facing, target_facing))

        self._update_markers(torso_track_state.pos, current_facing,
                             target_facing)

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
            ('current_facing', current_facing),
            ('desired_facing', target_facing),
            ('angle_error', angle_error),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        angle_error = obs_dict['angle_error']
        upright = obs_dict[self._upright_obs_key]
        center_dist = np.linalg.norm(obs_dict['root_pos'][:2], axis=1)

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for closeness to desired facing direction.
            ('alignment_error_cost', -4 * angle_error),
            # Reward for closeness to center; i.e. being stationary.
            ('center_distance_cost', -4 * center_dist),
            # Bonus when mean error < 15deg or upright within 15deg.
            ('bonus_small', 5 * ((angle_error < 0.26) + (upright > 0.96))),
            # Bonus when error < 5deg and upright within 15deg.
            ('bonus_big', 10 * ((angle_error < 0.087) * (upright > 0.96))),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points', 1.0 - obs_dict['angle_error'] / np.pi),
            ('success', reward_dict['bonus_big'] > 0.0),
        ))

    def _update_markers(self, root_pos: np.ndarray, current_facing: np.ndarray,
                        target_facing: np.ndarray):
        """Updates the simulation markers denoting the facing direction."""
        self.model.body_pos[self._markers_bid][:2] = root_pos[:2]
        current_angle = np.arctan2(current_facing[1], current_facing[0])
        target_angle = np.arctan2(target_facing[1], target_facing[0])

        self.model.body_quat[self._current_angle_bid] = euler2quat(
            0, 0, current_angle, axes='rxyz')
        self.model.body_quat[self._target_angle_bid] = euler2quat(
            0, 0, target_angle, axes='rxyz')
        self.sim.forward()


@configurable(pickleable=True)
class DKittyOrientFixed(BaseDKittyOrient):
    """Stand up from a fixed position."""

    def _reset(self):
        """Resets the environment."""
        # Put target behind the D'Kitty (180deg rotation).
        self._initial_angle = 0
        self._target_angle = np.pi
        super()._reset()


@configurable(pickleable=True)
class DKittyOrientRandom(BaseDKittyOrient):
    """Walk straight towards a random location."""

    def __init__(
            self,
            *args,
            # +/-60deg
            initial_angle_range: Tuple[float, float] = (-np.pi / 3, np.pi / 3),
            # 180 +/- 60deg
            target_angle_range: Tuple[float, float] = (2 * np.pi / 3,
                                                       4 * np.pi / 3),
            **kwargs):
        """Initializes the environment.

        Args:
            initial_angle_range: The range to sample an initial orientation
                of the D'Kitty about the z-axis.
            target_angle_range: The range to sample a target orientation of
                the D'Kitty about the z-axis.
        """
        super().__init__(*args, **kwargs)
        self._initial_angle_range = initial_angle_range
        self._target_angle_range = target_angle_range

    def _reset(self):
        """Resets the environment."""
        self._initial_angle = self.np_random.uniform(*self._initial_angle_range)
        self._target_angle = self.np_random.uniform(*self._target_angle_range)
        super()._reset()


@configurable(pickleable=True)
class DKittyOrientRandomDynamics(DKittyOrientRandom):
    """Walk straight towards a random location."""

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
            kp_range=(2.8, 3.2),
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
