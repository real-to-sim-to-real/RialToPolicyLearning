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

"""Shared logic for all DClaw environments."""

import abc
import collections
from typing import Dict, Optional, Sequence, Union

import gym
import numpy as np

from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_utils import CalibrationMap
from robel.dclaw import scripted_reset
from robel.robot_env import make_box_space, RobotEnv

# Convenience constants.
PI = np.pi

# Threshold near the joint limits at which we consider to be unsafe.
SAFETY_POS_THRESHOLD = 5 * PI / 180  # 5 degrees

SAFETY_VEL_THRESHOLD = 1.0  # 1rad/s

# Current threshold above which we consider as unsafe.
SAFETY_CURRENT_THRESHOLD = 200  # mA

# Mapping of motor ID to (scale, offset).
DEFAULT_DCLAW_CALIBRATION_MAP = CalibrationMap({
    # Finger 1
    10: (1, -PI / 2),
    11: (1, -PI),
    12: (1, -PI),
    # Finger 2
    20: (1, -PI / 2),
    21: (1, -PI),
    22: (1, -PI),
    # Finger 3
    30: (1, -PI / 2),
    31: (1, -PI),
    32: (1, -PI),
    # Object
    50: (1, -PI),
    # Guide
    60: (1, -PI),
})


class BaseDClawEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks."""

    def __init__(self,
                 *args,
                 device_path: Optional[str] = None,
                 sim_observation_noise: Optional[float] = None,
                 **kwargs):
        """Initializes the environment.

        Args:
            device_path: The device path to Dynamixel hardware.
            sim_observation_noise: If given, configures the RobotComponent to
                add noise to observations.
        """
        super().__init__(*args, **kwargs)
        self._device_path = device_path
        self._sim_observation_noise = sim_observation_noise

        # Create the robot component.
        robot_builder = RobotComponentBuilder()
        self._configure_robot(robot_builder)
        self.robot = self._add_component(robot_builder)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        state = self.robot.get_state('dclaw')
        return {'qpos': state.qpos, 'qvel': state.qvel}

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state(
            {'dclaw': RobotState(qpos=state['qpos'], qvel=state['qvel'])})

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        # Add the main D'Claw group.
        builder.add_group(
            'dclaw',
            qpos_indices=range(9),
            qpos_range=[
                (-0.48, 0.48),  # ~27.5 degrees for top servos.
                (-PI / 3, PI / 3),  # 60 degrees for middle servos.
                (-PI / 2, PI / 2),  # 90 degrees for bottom servos.
            ] * 3,
            qvel_range=[(-2 * PI / 3, 2 * PI / 3)] * 9)
        if self._sim_observation_noise is not None:
            builder.update_group(
                'dclaw', sim_observation_noise=self._sim_observation_noise)
        # If a device path is given, set the motor IDs and calibration map.
        if self._device_path:
            builder.set_dynamixel_device_path(self._device_path)
            builder.set_hardware_calibration_map(DEFAULT_DCLAW_CALIBRATION_MAP)
            builder.update_group(
                'dclaw', motor_ids=[10, 11, 12, 20, 21, 22, 30, 31, 32])
            scripted_reset.add_groups_for_reset(builder)

    def _initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dclaw').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size,))

    def _get_safety_scores(
            self,
            pos: Optional[np.ndarray] = None,
            vel: Optional[np.ndarray] = None,
            current: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Computes safety-related scores for D'Claw robots.

        Args:
            pos: The joint positions.
            vel: The joint velocities.
            current: The motor currents.

        Returns:
            A dictionary of safety scores for the given values.
        """
        scores = collections.OrderedDict()
        dclaw_config = self.robot.get_config('dclaw')

        if pos is not None and dclaw_config.qpos_range is not None:
            # Calculate lower and upper separately so broadcasting works when
            # positions are batched.
            near_lower_limit = (
                np.abs(dclaw_config.qpos_range[:, 0] - pos) <
                SAFETY_POS_THRESHOLD)
            near_upper_limit = (
                np.abs(dclaw_config.qpos_range[:, 1] - pos) <
                SAFETY_POS_THRESHOLD)
            near_pos_limit = np.sum(near_lower_limit | near_upper_limit, axis=1)
            scores['safety_pos_violation'] = near_pos_limit

        if vel is not None:
            above_vel_limit = np.sum(np.abs(vel) > SAFETY_VEL_THRESHOLD, axis=1)
            scores['safety_vel_violation'] = above_vel_limit

        if current is not None:
            above_current_limit = np.sum(
                np.abs(current) > SAFETY_CURRENT_THRESHOLD, axis=1)
            scores['safety_current_violation'] = above_current_limit
        return scores


class BaseDClawObjectEnv(BaseDClawEnv, metaclass=abc.ABCMeta):
    """Base environment for all DClaw robot tasks with objects."""

    def __init__(self, *args, use_guide: bool = False, **kwargs):
        """Initializes the environment.

        Args:
            use_guide: If True, activates an object motor in hardware to use
                to show the goal.
        """
        self._use_guide = use_guide
        super().__init__(*args, **kwargs)

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        claw_state, object_state = self.robot.get_state(['dclaw', 'object'])
        return {
            'claw_qpos': claw_state.qpos,
            'claw_qvel': claw_state.qvel,
            'object_qpos': object_state.qpos,
            'object_qvel': object_state.qvel,
        }

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state({
            'dclaw': RobotState(
                qpos=state['claw_qpos'], qvel=state['claw_qvel']),
            'object': RobotState(
                qpos=state['object_qpos'], qvel=state['object_qvel']),
        })

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        super()._configure_robot(builder)
        # Add the object group.
        builder.add_group(
            'object',
            qpos_indices=[-1],  # The object is the last qpos.
            qpos_range=[(-PI, PI)])
        if self._sim_observation_noise is not None:
            builder.update_group(
                'object', sim_observation_noise=self._sim_observation_noise)
        if self._device_path:
            builder.update_group('object', motor_ids=[50])

        # Add the guide group, which is a no-op if the guide motor is unused.
        builder.add_group('guide')
        if self._use_guide and self._device_path:
            builder.update_group('guide', motor_ids=[60], use_raw_actions=True)

    def _reset_dclaw_and_object(
            self,
            claw_pos: Optional[Sequence[float]] = None,
            claw_vel: Optional[Sequence[float]] = None,
            object_pos: Optional[Union[float, Sequence[float]]] = None,
            object_vel: Optional[Union[float, Sequence[float]]] = None,
            guide_pos: Optional[Union[float, Sequence[float]]] = None):
        """Reset procedure for DClaw robots that manipulate objects.

        Args:
            claw_pos: The joint positions for the claw (radians).
            claw_vel: The joint velocities for the claw (radians/second). This
                is ignored on hardware.
            object_pos: The joint position for the object (radians).
            object_vel: The joint velocity for the object (radians/second). This
                is ignored on hardware.
            guide_pos: The joint position for the guide motor (radians). The
                guide motor is optional for marking the desired position.
        """
        # Set defaults if parameters are not given.
        claw_init_state, object_init_state = self.robot.get_initial_state(
            ['dclaw', 'object'])
        claw_pos = (
            claw_init_state.qpos if claw_pos is None else np.asarray(claw_pos))
        claw_vel = (
            claw_init_state.qvel if claw_vel is None else np.asarray(claw_vel))
        object_pos = (
            object_init_state.qpos
            if object_pos is None else np.atleast_1d(object_pos))
        object_vel = (
            object_init_state.qvel
            if object_vel is None else np.atleast_1d(object_vel))
        guide_pos = (
            np.zeros(1) if guide_pos is None else np.atleast_1d(guide_pos))

        if self.robot.is_hardware:
            scripted_reset.reset_to_states(
                self.robot, {
                    'dclaw': RobotState(qpos=claw_pos),
                    'object': RobotState(qpos=object_pos),
                    'guide': RobotState(qpos=guide_pos),
                })
        else:
            self.robot.set_state({
                'dclaw': RobotState(qpos=claw_pos, qvel=claw_vel),
                'object': RobotState(qpos=object_pos, qvel=object_vel),
            })
