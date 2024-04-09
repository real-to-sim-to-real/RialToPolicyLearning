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

"""Puash task.

The robot must push a block to a target point.
By default, this environment does not employ any randomizations, and so will
exhibit poor sim2real transfer. For better transfer, one may want to borrow
some of the randomizations used in DKittyWalkRandomDynamics.
"""

import collections
from typing import Dict, Optional, Sequence

import numpy as np

from robel.components.tracking import TrackerState
from robel.dkitty.avoid import DKittyAvoid
from robel.utils.math_utils import calculate_cosine

DKITTY_ASSET_PATH = 'robel/dkitty/assets/dkitty_push-v0.xml'

DEFAULT_OBSERVATION_KEYS = (
    'root_pos',
    'root_euler',
    'kitty_qpos',
    'root_vel',
    'root_angular_vel',
    'last_action',
    'upright',
    'heading',
    'block_error',
    'target_error',
)


class DKittyPush(DKittyAvoid):
    """Push task."""

    def __init__(
            self,
            asset_path: str = DKITTY_ASSET_PATH,
            observation_keys: Sequence[str] = DEFAULT_OBSERVATION_KEYS,
            device_path: Optional[str] = None,
            frame_skip: int = 40,
            upright_threshold: float = 0.9,  # cos(~25deg)
            upright_reward: float = 1,
            falling_reward: float = 0,  # Don't penalize falling.
            **kwargs):
        """Initializes the environment.

        Args:
            See DKittyAvoid.
        """
        super().__init__(
            asset_path=asset_path,
            observation_keys=observation_keys,
            device_path=device_path,
            frame_skip=frame_skip,
            upright_threshold=upright_threshold,
            upright_reward=upright_reward,
            falling_reward=falling_reward,
            **kwargs)

    def _reset(self):
        """Resets the environment."""
        target_dist = self.np_random.uniform(low=1.5, high=2.5)
        target_theta = np.pi / 2 + self.np_random.uniform(low=-1, high=1)
        self._initial_target_pos = target_dist * np.array([
            np.cos(target_theta), np.sin(target_theta), 0
        ])

        for _ in range(10):
            block_dist = self.np_random.uniform(low=0.6, high=1.2)
            block_theta = np.pi / 2 + self.np_random.uniform(
                low=np.pi / 3, high=5 * np.pi / 3)
            block_x = (
                target_dist * np.cos(target_theta) +
                block_dist * np.cos(block_theta))
            block_y = (
                target_dist * np.sin(target_theta) +
                block_dist * np.sin(block_theta))
            # Check that block is at least 0.5 away from initial robot point (0, 0).
            if np.linalg.norm([block_x, block_y]) > 0.5:
                break
        self._initial_block_pos = np.array([block_x, block_y, 0.2])

        self._reset_dkitty_standing()

        target_pos = self._initial_target_pos
        heading_pos = self._initial_heading_pos
        if heading_pos is None:
            heading_pos = target_pos
        block_pos = self._initial_block_pos

        # Set the tracker locations.
        self.tracker.set_state({
            'torso': TrackerState(pos=np.zeros(3), rot=np.identity(3)),
            'target': TrackerState(pos=target_pos),
            'heading': TrackerState(pos=heading_pos),
            'block': TrackerState(pos=block_pos),
        })

    def get_obs_dict(self) -> Dict[str, np.ndarray]:
        """Returns the current observation of the environment.

        Returns:
            A dictionary of observation values. This should be an ordered
            dictionary if `observation_keys` isn't set.
        """
        robot_state = self.robot.get_state('dkitty')
        (target_state, heading_state, block_state,
         torso_track_state) = self.tracker.get_state(
             ['target', 'heading', 'block', 'torso'])

        target_xy = target_state.pos[:2]
        block_xy = block_state.pos[:2]
        kitty_xy = torso_track_state.pos[:2]

        # Get the heading of the torso (the x-axis).
        current_heading = torso_track_state.rot[:2, 0]

        # Get the direction towards the heading location.
        desired_heading = heading_state.pos[:2] - kitty_xy

        # Calculate the alignment of the heading with the desired direction.
        heading = calculate_cosine(current_heading, desired_heading)

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
            ('heading', heading),
            ('target_pos', target_xy),
            ('block_error', block_xy - kitty_xy),
            ('target_error', target_xy - kitty_xy),
        ))

    def get_reward_dict(
            self,
            action: np.ndarray,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        target_xy_dist = np.linalg.norm(obs_dict['block_error'] -
                                        obs_dict['target_error'])
        block_xy_dist = np.linalg.norm(obs_dict['block_error'])
        bonus = 20 * (target_xy_dist < 0.5)

        reward_dict = collections.OrderedDict((
            # Add reward terms for being upright.
            *self._get_upright_rewards(obs_dict).items(),
            # Reward for proximity of block to the target.
            ('target_dist_cost', -4 * target_xy_dist),
            # Reward for proximity of agent to the block.
            ('block_dist_cost', -2 * block_xy_dist),
            # Reached target bonus.
            ('bonus', bonus),
        ))
        return reward_dict

    def get_score_dict(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns a standardized measure of success for the environment."""
        return collections.OrderedDict((
            ('points',
             -np.linalg.norm(obs_dict['block_error'] - obs_dict['target_error'])
            ),
            ('success', reward_dict['bonus'] > 0.0),
        ))
