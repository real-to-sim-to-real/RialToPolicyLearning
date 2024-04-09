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

import numpy as np

from robel.components.builder import ComponentBuilder
from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_robot import DynamixelRobotComponent
from robel.components.tracking import TrackerComponentBuilder
from robel.components.tracking.tracker import TrackerComponent
from robel.utils.reset_procedure import ResetProcedure

# Maximum values for each joint.
BASEMAX = .8
MIDMAX = 2.4
FOOTMAX = 2.5

# Common parameters for all `set_state` commands.
SET_PARAMS = dict(
    error_tol=5 * np.pi / 180,  # 5 degrees
    last_diff_tol=.1 * np.pi / 180,  # 5 degrees
)

# Convenience constants.
PI = np.pi
PI2 = np.pi / 2
PI4 = np.pi / 4
PI6 = np.pi / 6

OUTWARD_TUCK_POSE = np.array([0, -MIDMAX, FOOTMAX, 0, MIDMAX, -FOOTMAX])
INWARD_TUCK_POSE = np.array([0, MIDMAX, -FOOTMAX, 0, -MIDMAX, FOOTMAX])


class ScriptedDKittyResetProcedure(ResetProcedure):
    """Scripted reset procedure for D'Kitty.

    This resets the D'Kitty to a standing position.
    """
    def __init__(self,
                 upright_threshold: float = 0.9,
                 standing_height: float = 0.35,
                 height_tolerance: float = 0.05,
                 max_attempts: int = 5):
        super().__init__()
        self._upright_threshold = upright_threshold
        self._standing_height = standing_height
        self._height_tolerance = height_tolerance
        self._max_attempts = max_attempts
        self._robot = None
        self._tracker = None

    def configure_reset_groups(self, builder: ComponentBuilder):
        """Configures the component groups needed for reset."""
        if isinstance(builder, RobotComponentBuilder):
            builder.add_group('left', motor_ids=[20, 21, 22, 30, 31, 32])
            builder.add_group('right', motor_ids=[10, 11, 12, 40, 41, 42])
            builder.add_group('front', motor_ids=[10, 11, 12, 20, 21, 22])
            builder.add_group('back', motor_ids=[30, 31, 32, 40, 41, 42])
        elif isinstance(builder, TrackerComponentBuilder):
            assert 'torso' in builder.group_configs

    def reset(self, robot: DynamixelRobotComponent, tracker: TrackerComponent):
        """Performs the reset procedure."""
        self._robot = robot
        self._tracker = tracker

        attempts = 0
        while not self._is_standing():
            attempts += 1
            if attempts > self._max_attempts:
                break

            if self._is_upside_down():
                self._perform_flip_over()

            self._perform_tuck_under()
            self._perform_stand_up()

    def _is_standing(self) -> bool:
        """Returns True if the D'Kitty is fully standing."""
        state = self._tracker.get_state('torso', raw_states=True)
        height = state.pos[2]
        upright = state.rot[2, 2]
        print('Upright: {:2f}, height: {:2f}'.format(upright, height))
        if upright < self._upright_threshold:
            return False
        if (np.abs(height - self._standing_height) > self._height_tolerance):
            return False
        return True

    def _get_uprightedness(self) -> float:
        """Returns the uprightedness of the D'Kitty."""
        return self._tracker.get_state('torso', raw_states=True).rot[2, 2]

    def _is_upside_down(self) -> bool:
        """Returns whether the D'Kitty is upside-down."""
        return self._get_uprightedness() < 0

    def _perform_flip_over(self):
        """Attempts to flip the D'Kitty over."""
        while self._is_upside_down():
            print('Is upside down {}; attempting to flip over...'.format(
                self._get_uprightedness()))
            # Spread flat and extended.
            self._perform_flatten()
            # If we somehow flipped over from that, we're done.
            if not self._is_upside_down():
                return
            # Tuck in one side while pushing down on the other side.
            self._robot.set_state(
                {
                    'left':
                        RobotState(qpos=np.array([-PI4, -MIDMAX, FOOTMAX] * 2)),
                    'right': RobotState(qpos=np.array([-PI - PI4, 0, 0] * 2)),
                },
                timeout=4,
                **SET_PARAMS,
            )
            # Straighten out the legs that were pushing down.
            self._robot.set_state(
                {
                    'left': RobotState(qpos=np.array([PI2, 0, 0] * 2)),
                    'right': RobotState(qpos=np.array([-PI2, 0, 0] * 2)),
                },
                timeout=4,
                **SET_PARAMS,
            )

    def _perform_tuck_under(self):
        """Tucks the D'Kitty's legs so that they're under itself."""
        # Bring in both sides of the D'Kitty while remaining flat.
        self._perform_flatten()
        # Tuck one side at a time.
        for side in ('left', 'right'):
            self._robot.set_state(
                {side: RobotState(qpos=np.array(INWARD_TUCK_POSE))},
                timeout=4,
                **SET_PARAMS,
            )

    def _perform_flatten(self):
        """Makes the D'Kitty go into a flat pose."""
        left_pose = INWARD_TUCK_POSE.copy()
        left_pose[[0, 3]] = PI2
        right_pose = INWARD_TUCK_POSE.copy()
        right_pose[[0, 3]] = -PI2
        self._robot.set_state(
            {
                'left': RobotState(qpos=left_pose),
                'right': RobotState(qpos=right_pose),
            },
            timeout=4,
            **SET_PARAMS,
        )

    def _perform_stand_up(self):
        """Makes the D'Kitty stand up."""
        # Flip the back and front.
        self._robot.set_state(
            {
                'back': RobotState(
                    qpos=np.array(OUTWARD_TUCK_POSE[3:].tolist() * 2)),
            },
            timeout=4,
            **SET_PARAMS,
        )
        self._robot.set_state(
            {
                'front': RobotState(
                    qpos=np.array(OUTWARD_TUCK_POSE[:3].tolist() * 2)),
            },
            timeout=4,
            **SET_PARAMS,
        )
        # Stand straight up.
        self._robot.set_state(
            {
                'dkitty': RobotState(qpos=np.zeros(12)),
            },
            timeout=3,
            **SET_PARAMS,
        )
        # Tuck a bit.
        self._robot.set_state(
            {
                'dkitty': RobotState(qpos=np.array([0, PI6, -PI6] * 4)),
            },
            timeout=1,
            **SET_PARAMS,
        )
        # Stand straight up.
        self._robot.set_state(
            {
                'dkitty': RobotState(qpos=np.zeros(12)),
            },
            timeout=3,
            **SET_PARAMS,
        )
