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

"""Hardware reset functions for the D'Kitty."""

import time

from robel.components.builder import ComponentBuilder
from robel.components.robot import RobotComponentBuilder
from robel.components.robot.dynamixel_robot import DynamixelRobotComponent
from robel.components.tracking import TrackerComponentBuilder
from robel.components.tracking.tracker import TrackerComponent
from robel.utils.reset_procedure import ResetProcedure


class ManualAutoDKittyResetProcedure(ResetProcedure):
    """Manual reset procedure for D'Kitty.

    This waits until the D'Kitty is placed upright and automatically starts the
    episode.
    """

    def __init__(self,
                 upright_threshold: float = 0.9,
                 max_height: float = 0.35,
                 min_successful_checks: int = 5,
                 check_interval_sec: float = 0.1,
                 print_interval_sec: float = 1.0,
                 episode_start_delay_sec: float = 1.0):
        super().__init__()
        self._upright_threshold = upright_threshold
        self._max_height = max_height
        self._min_successful_checks = min_successful_checks
        self._check_interval_sec = check_interval_sec
        self._print_interval_sec = print_interval_sec
        self._episode_start_delay_sec = episode_start_delay_sec
        self._last_print_time = 0
        self._robot = None
        self._tracker = None

    def configure_reset_groups(self, builder: ComponentBuilder):
        """Configures the component groups needed for reset."""
        if isinstance(builder, RobotComponentBuilder):
            assert 'dkitty' in builder.group_configs
        elif isinstance(builder, TrackerComponentBuilder):
            assert 'torso' in builder.group_configs

    def reset(self, robot: DynamixelRobotComponent, tracker: TrackerComponent):
        """Performs the reset procedure."""
        self._robot = robot
        self._tracker = tracker

    def finish(self):
        """Called when the reset is complete."""
        # Wait until the robot is sufficiently upright.
        self._wait_until_upright()

    def _wait_until_upright(self):
        """Waits until the D'Kitty is upright."""
        upright_checks = 0
        self._last_print_time = 0  # Start at 0 so print happens first time.
        while True:
            if self._is_dkitty_upright():
                upright_checks += 1
            else:
                upright_checks = 0
            if upright_checks > self._min_successful_checks:
                break
            time.sleep(self._check_interval_sec)

        print('Reset complete, starting episode...')
        time.sleep(self._episode_start_delay_sec)

    def _is_dkitty_upright(self) -> bool:
        """Checks if the D'Kitty is currently upright."""
        state = self._tracker.get_state('torso')
        height = state.pos[2]
        upright = state.rot[2, 2]

        cur_time = time.time()
        if cur_time - self._last_print_time >= self._print_interval_sec:
            self._last_print_time = cur_time
            print(('Waiting for D\'Kitty to be upright (upright: {:.2f}, '
                   'height: {:.2f})').format(upright, height))

        if upright < self._upright_threshold:
            return False
        if height > self._max_height:
            return False
        return True
