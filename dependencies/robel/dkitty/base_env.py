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

"""Shared logic for all DKitty environments."""

import abc
from typing import Dict, Optional, Sequence, Union

import gym
import numpy as np

from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_utils import CalibrationMap
from robel.components.tracking import (TrackerComponentBuilder,
                                            TrackerState, TrackerType)
from robel.dkitty.utils.manual_reset import ManualAutoDKittyResetProcedure
from robel.dkitty.utils.scripted_reset import ScriptedDKittyResetProcedure
from robel.robot_env import make_box_space, RobotEnv
from robel.utils.reset_procedure import ManualResetProcedure

# Convenience constants.
PI = np.pi

# Mapping of motor ID to (scale, offset).
DEFAULT_DKITTY_CALIBRATION_MAP = CalibrationMap({
    # Front right leg.
    10: (1, -3. * PI / 2),
    11: (-1, PI),
    12: (-1, PI),
    # Front left leg.
    20: (1, -PI / 2),
    21: (1, -PI),
    22: (1, -PI),
    # Back left leg.
    30: (-1, 3. * PI / 2),
    31: (1, -PI),
    32: (1, -PI),
    # Back right leg.
    40: (-1, PI / 2),
    41: (-1, PI),
    42: (-1, PI),
})


class BaseDKittyEnv(RobotEnv, metaclass=abc.ABCMeta):
    """Base environment for all DKitty robot tasks."""

    def __init__(self,
                 *args,
                 device_path: Optional[str] = None,
                 sim_observation_noise: Optional[float] = None,
                 reset_type: Optional[str] = None,
                 plot_tracking: bool = False,
                 phasespace_server: Optional[str] = None,
                 **kwargs):
        """Initializes the environment.

        Args:
            device_path: The device path to Dynamixel hardware.
            sim_observation_noise: If given, configures the RobotComponent to
                add noise to observations.
            manual_reset: If True, waits for the user to reset the robot
                instead of performing the automatic reset procedure.
            plot_tracking: If True, displays a plot that shows the tracked
                positions. NOTE: Currently this causes the environment to run
                more slowly.
            phasespace_server: The PhaseSpace server to connect to. If given,
                PhaseSpace is used as the tracking provider instead of OpenVR.
        """
        super().__init__(*args, **kwargs)
        self._device_path = device_path
        self._sim_observation_noise = sim_observation_noise

        # Configure the robot component.
        robot_builder = RobotComponentBuilder()
        self._configure_robot(robot_builder)

        # Configure the tracker component.
        tracker_builder = TrackerComponentBuilder()
        self._configure_tracker(tracker_builder)
        if phasespace_server:
            tracker_builder.set_tracker_type(
                TrackerType.PHASESPACE, server_address=phasespace_server)

        # Configure the hardware reset procedure.
        self._hardware_reset = None
        if self._device_path is not None:
            if reset_type is None or reset_type == 'scripted':
                self._hardware_reset = ScriptedDKittyResetProcedure()
            elif reset_type == 'manual-auto':
                self._hardware_reset = ManualAutoDKittyResetProcedure()
            elif reset_type == 'manual':
                self._hardware_reset = ManualResetProcedure()
            else:
                raise NotImplementedError(reset_type)
            for builder in (robot_builder, tracker_builder):
                self._hardware_reset.configure_reset_groups(builder)

        # Create the components.
        self.robot = self._add_component(robot_builder)
        self.tracker = self._add_component(tracker_builder)

        # Disable the constraint solver in hardware so that mimicked positions
        # do not participate in contact calculations.
        if self.robot.is_hardware:
            self.sim_scene.disable_option(constraint_solver=True)

        if plot_tracking and self.tracker.is_hardware:
            self.tracker.show_plot()

    def get_state(self) -> Dict[str, np.ndarray]:
        """Returns the current state of the environment."""
        kitty_state = self.robot.get_state('dkitty')
        torso_state = self.tracker.get_state('torso')
        return {
            'root_pos': torso_state.pos,
            'root_euler': torso_state.rot_euler,
            'root_vel': torso_state.vel,
            'root_angular_vel': torso_state.angular_vel,
            'kitty_qpos': kitty_state.qpos,
            'kitty_qvel': kitty_state.qvel,
        }

    def set_state(self, state: Dict[str, np.ndarray]):
        """Sets the state of the environment."""
        self.robot.set_state({
            'dkitty': RobotState(
                qpos=state['kitty_qpos'], qvel=state['kitty_qvel']),
        })
        self.tracker.set_state({
            'torso': TrackerState(
                pos=state['root_pos'],
                rot_euler=state['root_euler'],
                vel=state['root_vel'],
                angular_vel=state['root_angular_vel'])
        })

    def _configure_robot(self, builder: RobotComponentBuilder):
        """Configures the robot component."""
        builder.add_group(
            'dkitty',
            actuator_indices=range(12),
            qpos_indices=range(6, 18),
            qpos_range=[
                # FR
                (-0.5, 0.279),
                (0.0, PI / 2),
                (-2.0, 0.0),
                # FL
                (-0.279, 0.5),
                (0.0, PI / 2),
                (-2.0, 0.0),
                # BL
                (-0.279, 0.5),
                (0.0, PI / 2),
                (-2.0, 0.0),
                # BR
                (-0.5, 0.279),
                (0.0, PI / 2),
                (-2.0, 0.0),
            ],
            qvel_range=[(-PI, PI)] * 12,
        )
        if self._sim_observation_noise is not None:
            builder.update_group(
                'dkitty', sim_observation_noise=self._sim_observation_noise)
        # If a device path is given, set the motor IDs and calibration map.
        if self._device_path is not None:
            builder.set_dynamixel_device_path(self._device_path)
            builder.set_hardware_calibration_map(DEFAULT_DKITTY_CALIBRATION_MAP)
            builder.update_group(
                'dkitty',
                motor_ids=[10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42])

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""

    def _initialize_action_space(self) -> gym.Space:
        """Returns the observation space to use for this environment."""
        qpos_indices = self.robot.get_config('dkitty').qpos_indices
        return make_box_space(-1.0, 1.0, shape=(qpos_indices.size,))

    def _reset_dkitty_standing(self,
                               kitty_pos: Optional[Sequence[float]] = None,
                               kitty_vel: Optional[Sequence[float]] = None):
        """Resets the D'Kitty to a standing position.

        Args:
            kitty_pos: The joint positions (radians).
            kitty_vel: The joint velocities (radians/second).
        """
        # Set defaults if parameters are not given.
        kitty_pos = np.zeros(12) if kitty_pos is None else np.asarray(kitty_pos)
        kitty_vel = np.zeros(12) if kitty_vel is None else np.asarray(kitty_vel)

        # Perform the scripted reset if we're not doing manual resets.
        if self._hardware_reset:
            self._hardware_reset.reset(robot=self.robot, tracker=self.tracker)

        # Reset the robot state.
        self.robot.set_state({
            'dkitty': RobotState(qpos=kitty_pos, qvel=kitty_vel),
        })

        # Complete the hardware reset.
        if self._hardware_reset:
            self._hardware_reset.finish()

        # Reset the clock to 0 for hardware.
        if self.robot.is_hardware:
            self.robot.reset_time()


class BaseDKittyUprightEnv(BaseDKittyEnv):
    """Base environment for D'Kitty tasks where the D'Kitty must be upright."""

    def __init__(
            self,
            *args,
            torso_tracker_id: Optional[Union[str, int]] = None,
            upright_obs_key: str = 'upright',
            upright_threshold: float = 0,  # cos(90deg).
            upright_reward: float = 1,
            falling_reward: float = -100,
            **kwargs):
        """Initializes the environment.

        Args:
            torso_tracker_id: The device index or serial of the tracking device
                for the D'Kitty torso.
            upright_obs_key: The observation key for uprightnedness.
            upright_threshold: The threshold (in [0, 1]) above which the D'Kitty
                is considered to be upright. If the cosine similarity of the
                D'Kitty's z-axis with the global z-axis is below this threshold,
                the D'Kitty is considered to have fallen.
            upright_reward: The reward multiplier for uprightedness.
            falling_reward: The reward multipler for falling.
            **kwargs: Arguemnts to pass to BaseDKittyEnv.
        """
        self._torso_tracker_id = torso_tracker_id
        super().__init__(*args, **kwargs)

        self._upright_obs_key = upright_obs_key
        self._upright_threshold = upright_threshold
        self._upright_reward = upright_reward
        self._falling_reward = falling_reward

    def _configure_tracker(self, builder: TrackerComponentBuilder):
        """Configures the tracker component."""
        super()._configure_tracker(builder)
        builder.add_tracker_group(
            'torso',
            hardware_tracker_id=self._torso_tracker_id,
            sim_params=dict(
                element_name='torso',
                element_type='joint',
                qpos_indices=range(6),
            ),
            hardware_params=dict(
                is_origin=True,
                # tracked_rotation_offset=(-1.57, 0, 1.57),
            ))

    def _get_upright_obs(self,
                         torso_track_state: TrackerState) -> Dict[str, float]:
        """Returns a dictionary of uprightedness observations."""
        return {self._upright_obs_key: torso_track_state.rot[2, 2]}

    def _get_upright_rewards(
            self,
            obs_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Returns the reward for the given action and observation."""
        upright = obs_dict[self._upright_obs_key]
        return {
            'upright': (
                self._upright_reward * (upright - self._upright_threshold) /
                (1 - self._upright_threshold)),
            'falling': self._falling_reward *
                       (upright < self._upright_threshold),
        }

    def get_done(
            self,
            obs_dict: Dict[str, np.ndarray],
            reward_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Returns whether the episode should terminate."""
        return obs_dict[self._upright_obs_key] < self._upright_threshold
