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

"""Base logic for hardware robots."""

import abc
import logging
import time
from typing import Iterable, Optional, Tuple

import numpy as np

from robel.components.robot.group_config import RobotGroupConfig
from robel.components.robot.robot import RobotComponent, RobotState

# Default tolerance for determining if the hardware has reached a state.
DEFAULT_ERROR_TOL = 1. * np.pi / 180


class HardwareRobotGroupConfig(RobotGroupConfig):
    """Stores group configuration for a HardwareRobotComponent."""

    def __init__(self,
                 *args,
                 calib_scale: Optional[Iterable[float]] = None,
                 calib_offset: Optional[Iterable[float]] = None,
                 **kwargs):
        """Initializes a new configuration for a HardwareRobotComponent group.

        Args:
            calib_scale: A scaling factor that is multipled with state to
                convert from component state space to hardware state space,
                and divides control to convert from hardware control space to
                component control space.
            calib_offset: An offset that is added to state to convert from
                component state space to hardware state space, and subtracted
                from control to convert from hardware control space to
                component control space.
        """
        super().__init__(*args, **kwargs)

        self.calib_scale = None
        if calib_scale is not None:
            self.calib_scale = np.array(calib_scale, dtype=np.float32)

        self.calib_offset = None
        if calib_offset is not None:
            self.calib_offset = np.array(calib_offset, dtype=np.float32)


class HardwareRobotComponent(RobotComponent, metaclass=abc.ABCMeta):
    """Base component for hardware robots."""

    def __init__(self, *args, **kwargs):
        """Initializes the component."""
        super().__init__(*args, **kwargs)
        self.reset_time()

    @property
    def is_hardware(self) -> bool:
        """Returns True if this is a hardware component."""
        return True

    @property
    def time(self) -> float:
        """Returns the time (total sum of timesteps) since the last reset."""
        return self._time

    def reset_time(self):
        """Resets the timer for the component."""
        self._last_reset_time = time.time()
        self._time = 0

    def _process_group(self, **config_kwargs) -> HardwareRobotGroupConfig:
        """Processes the configuration for a group."""
        return HardwareRobotGroupConfig(self.sim_scene, **config_kwargs)

    def _calibrate_state(self, state: RobotState,
                         group_config: HardwareRobotGroupConfig):
        """Converts the given state from hardware space to component space."""
        # Calculate qpos' = qpos * scale + offset, and qvel' = qvel * scale.
        if group_config.calib_scale is not None:
            assert state.qpos.shape == group_config.calib_scale.shape
            assert state.qvel.shape == group_config.calib_scale.shape
            state.qpos *= group_config.calib_scale
            state.qvel *= group_config.calib_scale
        if group_config.calib_offset is not None:
            assert state.qpos.shape == group_config.calib_offset.shape
            # Only apply the offset to positions.
            state.qpos += group_config.calib_offset

    def _decalibrate_qpos(self, qpos: np.ndarray,
                          group_config: HardwareRobotGroupConfig) -> np.ndarray:
        """Converts the given position from component to hardware space."""
        # Calculate qpos' = (qpos - offset) / scale.
        if group_config.calib_offset is not None:
            assert qpos.shape == group_config.calib_offset.shape
            qpos = qpos - group_config.calib_offset
        if group_config.calib_scale is not None:
            assert qpos.shape == group_config.calib_scale.shape
            qpos = qpos / group_config.calib_scale
        return qpos

    def _synchronize_timestep(self, minimum_sleep: float = 1e-4):
        """Waits for one timestep to elapse."""
        # Block the thread such that we've waited at least `step_duration` time
        # since the last call to `_synchronize_timestep`.
        time_since_reset = time.time() - self._last_reset_time
        elapsed_time = time_since_reset - self._time
        remaining_step_time = self.sim_scene.step_duration - elapsed_time
        if remaining_step_time > minimum_sleep:
            time.sleep(remaining_step_time)
        elif remaining_step_time < 0:
            logging.warning('Exceeded timestep by %0.4fs', -remaining_step_time)

        # Update the current time, relative to the last reset time.
        self._time = time.time() - self._last_reset_time

    def _wait_for_desired_states(
            self,
            desired_states: Iterable[Tuple[RobotGroupConfig, RobotState]],
            error_tol: float = DEFAULT_ERROR_TOL,
            timeout: float = 3.0,
            poll_interval: float = 0.25,
            initial_sleep: Optional[float] = 0.25,
            last_diff_tol: Optional[float] = DEFAULT_ERROR_TOL,
            last_diff_ticks: int = 2,
    ):
        """Polls the current state until it reaches the desired state.

        Args:
            desired_states: The desired states to wait for.
            error_tol: The maximum position difference within which the desired
                state is considered to have been reached.
            timeout: The maximum amount of time to wait, in seconds.
            poll_interval: The interval in seconds to poll the current state.
            initial_sleep: The initial time to sleep before polling.
            last_diff_tol: The maximum position difference between the current
                state and the last state at which motion is considered to be
                stopped, thus waiting will terminate early.
            last_diff_ticks: The number of cycles where the last difference
                tolerance check must pass for waiting to terminate early.
        """

        # Define helper function to compare two state sets.
        def all_states_close(states_a, states_b, tol):
            all_close = True
            for state_a, state_b in zip(states_a, states_b):
                if not np.allclose(state_a.qpos, state_b.qpos, atol=tol):
                    all_close = False
                    break
            return all_close

        # Poll for the hardware move command to complete.
        configs, desired_states = zip(*desired_states)
        previous_states = None
        ticks_until_termination = last_diff_ticks
        start_time = time.time()

        if initial_sleep is not None and initial_sleep > 0:
            time.sleep(initial_sleep)

        while True:
            cur_states = self._get_group_states(configs)
            # Terminate if the current states have reached the desired states.
            if all_states_close(cur_states, desired_states, tol=error_tol):
                return
            # Terminate if the current state and previous state are the same.
            # i.e. the robot is unable to move further.
            if previous_states is not None and all_states_close(
                    cur_states, previous_states, tol=last_diff_tol):
                if not ticks_until_termination:
                    logging.warning(
                        'Robot stopped motion; terminating wait early.')
                    return
                ticks_until_termination -= 1
            else:
                ticks_until_termination = last_diff_ticks

            if time.time() - start_time > timeout:
                logging.warning('Reset timed out after %1.1fs', timeout)
                return
            previous_states = cur_states
            time.sleep(poll_interval)

    def _copy_to_simulation_state(
            self, group_states: Iterable[Tuple[RobotGroupConfig, RobotState]]):
        """Copies the given states to the simulation."""
        for config, state in group_states:
            # Skip if this is a hardware-only group.
            if config.qpos_indices is None:
                continue
            if state.qpos is not None:
                self.sim_scene.data.qpos[config.qpos_indices] = state.qpos
            if state.qvel is not None:
                self.sim_scene.data.qvel[config.qvel_indices] = state.qvel

        # Recalculate forward dynamics.
        self.sim_scene.sim.forward()
        self.sim_scene.renderer.refresh_window()
