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

"""Script to test resetting robot hardware environments.

To run:
python -m robel.scripts.reset_hardware \
    -e DKittyWalkFixed-v0 -d /dev/ttyUSB0
"""

import argparse
import logging
import time

import gym

import robel
from robel.scripts.utils import parse_env_args


def main():
    # Get command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n',
        '--num_repeats',
        type=int,
        default=1,
        help='The number of resets to perform.')
    env_id, params, args = parse_env_args(parser)

    # Show INFO-level logs.
    logging.basicConfig(level=logging.INFO)

    # Create the environment and get the robot component.
    robel.set_env_params(env_id, params)
    env = gym.make(env_id).unwrapped
    assert env.robot.is_hardware

    for i in range(args.num_repeats):
        print('Starting reset #{}'.format(i))

        # Disengage all of the motors and let the dkitty fall.
        env.robot.set_motors_engaged(None, engaged=False)

        print('Place the robot to a starting position.')
        input('Press Enter to start the reset...')

        # Start with all motors engaged.
        env.robot.set_motors_engaged(None, engaged=True)
        env.reset()

        print('Done reset! Turning off the robot in a few seconds.')
        time.sleep(2)


if __name__ == '__main__':
    main()
