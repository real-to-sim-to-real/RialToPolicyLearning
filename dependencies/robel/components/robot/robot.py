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

"""Component implementation for reading and writing data to/from robots.

This abstracts differences between a MuJoCo simulation and a hardware robot.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from robel.components.base import BaseComponent
from robel.components.robot.group_config import (ControlMode,
                                                      RobotGroupConfig)


class RobotState:
    """Data class that represents the state of the robot."""

    def __init__(self,
                 qpos: Optional[np.ndarray] = None,
                 qvel: Optional[np.ndarray] = None,
                 qacc: Optional[np.ndarray] = None):
        """Initializes a new state object.

        Args:
            qpos: The joint positions, in generalized joint coordinates.
            qvel: The velocities for each degree of freedom.
            qacc: The acceleration for each degree of freedom. This is returned
                by `get_state`. `reset` will ignore this property.
        """
        self.qpos = qpos
        self.qvel = qvel
        self.qacc = qacc


class RobotComponent(BaseComponent):
    """Component for reading sensor data and actuating robots."""

    @property
    def time(self) -> float:
        """Returns the time (total sum of timesteps) since the last reset."""
        return self.sim_scene.data.time

    def _process_group(self, **config_kwargs) -> RobotGroupConfig:
        """Processes the configuration for a group."""
        return RobotGroupConfig(self.sim_scene, **config_kwargs)

    def step(self,
             control_groups: Dict[str, np.ndarray],
             denormalize: bool = True):
        """Runs one timestep of the robot for the given control.

        Examples:
            >>> robot.step({'dclaw': np.zeros(9)})

        Args:
            control_groups: A dictionary of control group name to desired
                control value to command the robot for a single timestep.
                e.g. for a control group with position control, the control
                value is an array of joint positions (in radians).
            denormalize: If True, denormalizes the action from normalized action
                space [-1, 1] to the robot's control space.
        """
        group_controls = []
        for group_name, control_values in control_groups.items():
            config = self.get_config(group_name)

            # Ignore if this is a hardware-only group.
            if not config.is_active:
                continue

            # Denormalize and enforce action bounds.
            if denormalize and not config.use_raw_actions:
                control_values = self._denormalize_action(
                    control_values, config)
            control_values = self._apply_action_bounds(control_values, config)

            group_controls.append((config, control_values))

        # Perform the control for all groups simultaneously.
        self._perform_timestep(group_controls)

    def set_state(self, state_groups: Dict[str, RobotState], **kwargs):
        """Moves the robot to the given initial state.

        Example:
            >>> robot.set_state({
            ...     'dclaw': RobotState(qpos=np.zeros(9), qvel=np.zeros(9)),
            ... })

        Args:
            state_groups: A mapping of control group name to desired position
                and velocity.
            **kwargs: Implementation-specific arguments.
        """
        group_states = []
        for group_name, state in state_groups.items():
            config = self.get_config(group_name)

            # Clip the position and velocity to the configured bounds.
            clipped_state = RobotState(qpos=state.qpos, qvel=state.qvel)
            if clipped_state.qpos is not None and config.qpos_range is not None:
                clipped_state.qpos = np.clip(clipped_state.qpos,
                                             config.qpos_range[:, 0],
                                             config.qpos_range[:, 1])
            if clipped_state.qvel is not None and config.qvel_range is not None:
                clipped_state.qvel = np.clip(clipped_state.qvel,
                                             config.qvel_range[:, 0],
                                             config.qvel_range[:, 1])

            group_states.append((config, clipped_state))

        # Set all states at once.
        self._set_group_states(group_states, **kwargs)

    def get_initial_state(
            self,
            groups: Union[str, Sequence[str]],
    ) -> Union[RobotState, Sequence[RobotState]]:
        """Returns the initial states for the given groups."""
        if isinstance(groups, str):
            configs = [self.get_config(groups)]
        else:
            configs = [self.get_config(name) for name in groups]
        states = []
        for config in configs:
            state = RobotState()
            # Return a blank state if this is a hardware-only group.
            if config.qpos_indices is None:
                states.append(state)
                continue

            state.qpos = self.sim_scene.init_qpos[config.qpos_indices].copy()
            state.qvel = self.sim_scene.init_qvel[config.qvel_indices].copy()
            states.append(state)
        if isinstance(groups, str):
            return states[0]
        return states

    def _get_group_states(
            self, configs: Sequence[RobotGroupConfig]) -> Sequence[RobotState]:
        """Returns the states for the given group configurations."""
        states = []
        for config in configs:
            state = RobotState()
            # Return a blank state if this is a hardware-only group.
            if config.qpos_indices is None:
                states.append(state)
                continue

            state.qpos = self.sim_scene.data.qpos[config.qpos_indices]
            state.qvel = self.sim_scene.data.qvel[config.qvel_indices]
            # qacc has the same dimensionality as qvel.
            state.qacc = self.sim_scene.data.qacc[config.qvel_indices]

            # Add observation noise to the state.
            self._apply_observation_noise(state, config)

            states.append(state)
        return states

    def _set_group_states(
            self, group_states: Sequence[Tuple[RobotGroupConfig, RobotState]]):
        """Sets the robot joints to the given states."""
        for config, state in group_states:
            if config.qpos_indices is None:
                continue
            if state.qpos is not None:
                self.sim_scene.data.qpos[config.qpos_indices] = state.qpos
            if state.qvel is not None:
                self.sim_scene.data.qvel[config.qvel_indices] = state.qvel

        self.sim_scene.sim.forward()

    def _perform_timestep(
            self,
            group_controls: Sequence[Tuple[RobotGroupConfig, np.ndarray]]):
        """Applies the given control values to the robot."""
        for config, control in group_controls:
            indices = config.actuator_indices
            assert len(indices) == len(control)
            self.sim_scene.data.ctrl[indices] = control

        # Advance the simulation by one timestep.
        self.sim_scene.advance()

    def _apply_observation_noise(self, state: RobotState,
                                 config: RobotGroupConfig):
        """Applies observation noise to the given state."""
        if config.sim_observation_noise is None or self.random_state is None:
            return

        # Define the noise calculation.
        def noise(value_range: np.ndarray):
            amplitude = config.sim_observation_noise * np.ptp(
                value_range, axis=1)
            return amplitude * self.random_state.uniform(
                low=-0.5, high=0.5, size=value_range.shape[0])

        if config.qpos_range is not None:
            state.qpos += noise(config.qpos_range)
        if config.qvel_range is not None:
            state.qvel += noise(config.qvel_range)

    def _denormalize_action(self, action: np.ndarray,
                            config: RobotGroupConfig) -> np.ndarray:
        """Denormalizes the given action."""
        if config.denormalize_center.shape != action.shape:
            raise ValueError(
                'Action shape ({}) does not match actuator shape: ({})'.format(
                    action.shape, config.denormalize_center.shape))
        assert config.denormalize_range is not None

        action = np.clip(action, -1.0, 1.0)
        return config.denormalize_center + (action * config.denormalize_range)

    def _apply_action_bounds(self, action: np.ndarray,
                             config: RobotGroupConfig) -> np.ndarray:
        """Clips the action using the given configuration.

        Args:
            action: The action to be applied to the robot.
            config: The group configuration that defines how the action is
                clipped.

        Returns:
            The clipped action.
        """
        if config.control_mode == ControlMode.JOINT_POSITION:
            # Apply position bounds.
            if config.qpos_range is not None:
                action = np.clip(action, config.qpos_range[:, 0],
                                 config.qpos_range[:, 1])

            # Apply velocity bounds.
            # NOTE: This uses the current simulation state to get the current
            # position. For hardware, this expects the hardware to update the
            # simulation state.
            if (config.qpos_indices is not None
                    and config.qvel_range is not None):
                # Calculate the desired velocity using the current position.
                cur_pos = self.sim_scene.data.qpos[config.qpos_indices]
                desired_vel = (
                    (action - cur_pos) / self.sim_scene.step_duration)
                # Clip with the velocity bounds.
                desired_vel = np.clip(desired_vel, config.qvel_range[:, 0],
                                      config.qvel_range[:, 1])
                action = cur_pos + desired_vel * self.sim_scene.step_duration

        elif config.control_mode == ControlMode.JOINT_VELOCITY:
            # Apply velocity bounds.
            if config.qvel_range is not None:
                action = np.clip(action, config.qvel_range[:, 0],
                                 config.qvel_range[:, 1])

        return action
