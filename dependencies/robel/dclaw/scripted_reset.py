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

"""Hardware reset functions for the D'Claw."""

import logging
import time
from typing import Dict

import numpy as np

from robel.components.robot import RobotComponentBuilder, RobotState
from robel.components.robot.dynamixel_robot import DynamixelRobotComponent

# Minimum time in seconds to wait before fully resetting the D'Claw.
MINIMUM_CLAW_RESET_TIME = 1.0

# The positional error tolerance to be considered as a good reset.
GOOD_ERROR_TOL = 5. * np.pi / 180.

# Maximum retries before quitting reset.
MAX_RESET_RETRIES = 3

# Reset position for top joints to disentangle.
TOP_DISENTANGLE_RESET_POS = np.array([np.pi / 4, np.pi / 4, np.pi / 4])

# Time to sleep in between disentangle interval steps.
DISENTANGLE_INTERVAL_TIME = 0.75


def add_groups_for_reset(builder: RobotComponentBuilder):
    """Defines groups required to perform the reset."""
    builder.add_group('dclaw_top', motor_ids=[10, 20, 30])
    builder.add_group('dclaw_middle', motor_ids=[11, 21, 31])
    builder.add_group('dclaw_bottom', motor_ids=[12, 22, 32])


def reset_to_states(robot: DynamixelRobotComponent,
                    states: Dict[str, RobotState]):
    """Resets the D'Claw to the given states.

    This is an multi-stage reset procedure that allows human intervention if
    motors are not resetting properly.

    Args:
        robot: The robot component to reset.
        states: The states to apply to the robot.
    """
    assert robot.is_hardware
    claw_state = states['dclaw']
    has_object = 'object' in robot.groups

    # Disable the top and bottom joints of the claw to help prevent tangling.
    robot.set_motors_engaged('dclaw_top', False)
    robot.set_motors_engaged('dclaw_bottom', False)
    robot.set_state({'dclaw': claw_state}, block=False)

    reset_start_time = time.time()

    # Reset the object and guide motors while the claw is moving.
    if has_object:
        robot.set_motors_engaged(['object', 'guide'], True)
        robot.set_state(
            {
                'object': states['object'],
                'guide': states['guide'],
            },
            timeout=10,
        )

    # Wait a minimum time before fully resetting the claw.
    reset_elapsed = time.time() - reset_start_time
    if reset_elapsed < MINIMUM_CLAW_RESET_TIME:
        time.sleep(MINIMUM_CLAW_RESET_TIME - reset_elapsed)

    # Fully reset the D'Claw.
    robot.set_motors_engaged('dclaw', True)
    robot.set_state({'dclaw': claw_state})

    # Check that the motors have actually reset.
    reset_retries = 0
    while True:
        cur_state = robot.get_state('dclaw')
        # Check positions one motor at a time for better diagnosing.
        bad_motors = []
        for i, motor_id in enumerate(robot.get_config('dclaw').motor_ids):
            if abs(cur_state.qpos[i] - claw_state.qpos[i]) > GOOD_ERROR_TOL:
                bad_motors.append(motor_id)

        if not bad_motors:
            break

        # Attempt to reset again.
        logging.error('[%d] Could not reset D\'Claw motors: %s', reset_retries,
                      str(bad_motors))
        reset_retries += 1

        # Wait for human assistance if too many resets have occurred.
        if reset_retries > MAX_RESET_RETRIES:
            print('\n' + '=' * 10)
            print('Please fix motors: {}'.format(bad_motors))
            print('=' * 10)
            input('Press Enter to resume.')
            reset_retries = 0

        # Try to disentangle.
        disentangle_dclaw(robot, claw_state.qpos)

        # Re-attempt the reset.
        robot.set_motors_engaged('dclaw', True)
        robot.set_state({'dclaw': claw_state})

    # Start the episode with the object disengaged.
    if has_object:
        robot.set_motors_engaged('object', False)
    robot.reset_time()


def disentangle_dclaw(robot: DynamixelRobotComponent, goal_pos: np.ndarray):
    """Performs a disentangling process to move to the given goal position."""
    assert goal_pos.shape == (9,)
    # Let the motors rest.
    robot.set_motors_engaged('dclaw', False)
    time.sleep(DISENTANGLE_INTERVAL_TIME)

    # Move the top joints upwards to free the lower joints.
    robot.set_motors_engaged('dclaw_top', True)
    robot.set_state({'dclaw_top': RobotState(qpos=TOP_DISENTANGLE_RESET_POS)},
                    block=False)
    time.sleep(DISENTANGLE_INTERVAL_TIME)

    # Reset the middle joints.
    robot.set_motors_engaged('dclaw_middle', True)
    robot.set_state({'dclaw_middle': RobotState(qpos=goal_pos[[1, 4, 7]])},
                    block=False)
    time.sleep(DISENTANGLE_INTERVAL_TIME)

    # Reset the lower joints.
    robot.set_motors_engaged('dclaw_bottom', True)
    robot.set_state({'dclaw_bottom': RobotState(qpos=goal_pos[[2, 5, 8]])},
                    block=False)
    time.sleep(DISENTANGLE_INTERVAL_TIME)
