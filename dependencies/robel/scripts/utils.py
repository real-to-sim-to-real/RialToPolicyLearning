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

"""Helper functions to load environments in scripts."""

import argparse
import collections
import csv
import os
import logging
from typing import Any, Dict, Optional, Sequence, Tuple

from gym.envs import registration as gym_reg
import numpy as np


class EpisodeLogger:
    """Logs episode data."""

    def __init__(self, csv_path: Optional[str] = None):
        """Creates a new logger.

        Args:
            csv_path: If given, saves episode data to a csv file.
        """
        self._file = None
        if csv_path:
            self._file = open(csv_path, 'w')
        self._writer = None
        self._episode_num = 0
        self._total_steps = 0

    def log_path(self,
                 path: Dict[str, Any],
                 reward_key: str = 'rewards',
                 env_info_key: str = 'infos'):
        """Logs the given path data as an episode."""
        self._total_steps += len(path[reward_key])
        total_reward = path[reward_key].sum()

        data = collections.OrderedDict((
            ('episode', self._episode_num),
            ('total_steps', self._total_steps),
            ('reward', total_reward),
        ))
        for info_key, info_values in path.get(env_info_key, {}).items():
            data[info_key + '-first-mean'] = np.mean(info_values[0])
            data[info_key + '-last-mean'] = np.mean(info_values[-1])
            data[info_key + '-mean-mean'] = np.mean(info_values)

        self.write_dict(data)
        self._episode_num += 1

    def write_dict(self, data: Dict[str, Any]):
        if self._file is None:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(data.keys()))
            self._writer.writeheader()
        self._writer.writerow(data)
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def parse_env_args(
        arg_parser: Optional[argparse.ArgumentParser] = None,
        default_env_name: Optional[str] = None,
) -> Tuple[str, Dict, argparse.Namespace]:
    """Parses the given arguments to get an environment ID and parameters.

    Args:
        arg_parser: An existing argument parser to add arguments to.

    Returns:
        env_name: The name of the environment.
        env_params: A dictionary of environment parameters that can be passed
            to the constructor of the environment class, or passed via
            `robel.set_env_params`.
        args: The Namespace object parsed by the ArgumentParser.
    """
    if arg_parser is None:
        arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '-e',
        '--env_name',
        required=(default_env_name is None),
        default=default_env_name,
        help='The environment to load.')
    arg_parser.add_argument('-d', '--device', help='The device to connect to.')
    arg_parser.add_argument(
        '--param',
        action='append',
        help=('A "key=value" pair to pass as an environment parameter. This '
              'be repeated, e.g. --param key1=val1 --param key2=val2'))
    arg_parser.add_argument(
        '--info', action='store_true', help='Turns on info logging.')
    arg_parser.add_argument(
        '--debug', action='store_true', help='Turns on debug logging.')
    args = arg_parser.parse_args()

    # Ensure the environment ID is valid.
    env_name = args.env_name
    if env_name not in gym_reg.registry.env_specs:
        raise ValueError('Unregistered environment ID: {}'.format(env_name))

    # Ensure the device exists, if given.
    device_path = args.device
    if device_path and not os.path.exists(device_path):
        raise ValueError('Device does not exist: {}'.format(device_path))

    # Parse environment params into a dictionary.
    env_params = {}
    if args.param:
        env_params = parse_env_params(args.param)

    if device_path:
        env_params['device_path'] = device_path

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    elif args.info:
        logging.basicConfig(level=logging.INFO)

    return env_name, env_params, args


def parse_env_params(user_entries: Sequence[str]) -> Dict[str, Any]:
    """Parses a list of `key=value` strings as a dictionary."""

    def is_value_convertable(v, convert_type) -> bool:
        try:
            convert_type(v)
        except ValueError:
            return False
        return True

    env_params = {}
    for user_text in user_entries:
        components = user_text.split('=')
        if len(components) != 2:
            raise ValueError('Key-values must be specified as `key=value`')
        value = components[1]
        if is_value_convertable(value, int):
            value = int(value)
        elif is_value_convertable(value, float):
            value = float(value)
        env_params[components[0]] = value
    return env_params
