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

"""Component implementation for interfacing with a Tracker."""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from transforms3d.quaternions import mat2quat

from robel.components.base import BaseComponent
from robel.components.tracking.group_config import TrackerGroupConfig


class TrackerState:
    """Data class that represents the state of the tracker."""

    def __init__(self,
                 pos: Optional[np.ndarray] = None,
                 rot: Optional[np.ndarray] = None,
                 rot_euler: Optional[np.ndarray] = None,
                 vel: Optional[np.ndarray] = None,
                 angular_vel: Optional[np.ndarray] = None):
        """Initializes a new state object.

        Args:
            pos: The (x, y, z) position of the tracked position. The z-axis
                points upwards.
            rot: The (3x3) rotation matrix of the tracked position.
            rot_euler: The (rx, ry, rz) Euler rotation of the tracked position.
            vel: The (x, y, z) velocity of the tracked position. This is only
                available for tracked joints.
            angular_vel: The (rx, ry, rz) angular velocity of the tracked
                position. This is only available for tracked joints.
        """
        self.pos = pos
        self.vel = vel
        self.angular_vel = angular_vel
        self._rot_mat = rot
        self._rot_euler = rot_euler

    @property
    def rot(self):
        """Returns the 3x3 rotation matrix."""
        if self._rot_mat is not None:
            return self._rot_mat
        if self._rot_euler is not None:
            self._rot_mat = euler2mat(*self._rot_euler, axes='rxyz')
        return self._rot_mat

    @rot.setter
    def rot(self, value):
        self._rot_mat = value

    @property
    def rot_euler(self):
        """Returns the (rx, ry, rz) Euler rotations."""
        if self._rot_euler is not None:
            return self._rot_euler
        if self._rot_mat is not None:
            self._rot_euler = mat2euler(self.rot, axes='rxyz')
        return self._rot_euler

    @rot_euler.setter
    def rot_euler(self, value):
        self._rot_euler = value

    @property
    def rot_quat(self):
        """Returns the rotation as a quaternion."""
        return mat2quat(self.rot)


class TrackerComponent(BaseComponent):
    """Component for reading tracking data."""

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return TrackerGroupConfig(self.sim_scene, **config_kwargs)

    def set_state(self, state_groups: Dict[str, TrackerState]):
        """Sets the tracker to the given initial state.

        Args:
            state_groups: A mapping of control group name to desired position
                and velocity.
        """
        changed_elements = []
        for group_name, state in state_groups.items():
            config = self.get_config(group_name)
            if self._set_element_state(state, config):
                changed_elements.append((config, state.pos))

        if changed_elements:
            self.sim_scene.sim.forward()
        # Verify that changes occured.
        for config, pos in changed_elements:
            new_pos = self._get_element_state(config).pos
            if new_pos is None or not np.allclose(new_pos, pos):
                logging.error(
                    'Element #%d is immutable (modify the XML with a non-zero '
                    'starting position).', config.element_name)

    def _get_group_states(
            self,
            configs: Sequence[TrackerGroupConfig],
    ) -> Sequence[TrackerState]:
        """Returns the TrackerState for the given groups.

        Args:
            configs: The group configurations to retrieve the states for.

        Returns:
            A list of TrackerState(timestamp, pos, quat, euler).
        """
        return [self._get_element_state(config) for config in configs]

    def _get_element_state(self, config: TrackerGroupConfig) -> TrackerState:
        """Returns the simulation element state for the given group config."""
        state = TrackerState()
        if config.element_id is None and config.qpos_indices is None:
            return state

        state.pos = config.get_pos(self.sim_scene)
        state.rot = config.get_rot(self.sim_scene)

        if config.qpos_indices is not None:
            state.vel = config.get_vel(self.sim_scene)
            state.angular_vel = config.get_angular_vel(self.sim_scene)

        if (config.sim_observation_noise is not None
                and self.random_state is not None):
            amplitude = config.sim_observation_noise / 2.0
            state.pos += self.random_state.uniform(
                low=-amplitude, high=amplitude, size=state.pos.shape)
        return state

    def _set_element_state(self,
                           state: TrackerState,
                           config: TrackerGroupConfig,
                           ignore_z_axis: bool = False) -> bool:
        """Sets the simulation state for the given element."""
        changed = False
        if config.element_id is None and config.qpos_indices is None:
            return changed
        if state.pos is not None:
            if ignore_z_axis:
                config.set_pos(self.sim_scene, state.pos[:2])
            else:
                config.set_pos(self.sim_scene, state.pos)
            changed = True
        if state.rot is not None:
            rot_quat = mat2quat(state.rot)
            config.set_rot_quat(self.sim_scene, rot_quat)
            changed = True
        return changed
