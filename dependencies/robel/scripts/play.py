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

"""Loads environments and allows the user to interact with it.

Example:
```
$> python robel.scripts.play
...
>>> load DClawTurnFixed-v0
>>> obs
claw_qpos: ...
claw_qvel: ...
...
>>> action [0]*9
Sending action: [0, 0, 0, 0, 0, 0, 0, 0, 0]
>>> reset
...
```

The script can be started with an environment:
```
$> python robel.scripts.play -e DClawTurnFixed-v0
```
"""

import collections
import cmd
import logging
import time
import threading
from typing import Any, Dict, Optional

import gym
import numpy as np

import robel
from robel.scripts.utils import parse_env_args

MIN_FRAME_TIME = 1.0 / 60.0

INTRODUCTION = """Interactive shell for Adept Environments.

Type `help` or `?` to list commands.
"""

_ENV_COMMANDS = ['obs', 'action', 'reset', 'random', 'engage', 'disengage']


class PlayShell(cmd.Cmd):
    """Implements a command-line interface for visualizing an environment."""

    intro = INTRODUCTION
    prompt = '>>> '

    def __init__(self,
                 env_name: Optional[str] = None,
                 env_params: Optional[Dict[str, Any]] = None):
        """Initializes a new command-line interface.

        Args:
            env_name: The environment to initially load.
            device: The device to run the environment on.
            env_params: Parameters to pass to the environment.
        """
        super().__init__()

        self._env_lock = threading.Lock()

        self._action = None
        self._do_random_actions = False
        self._do_reset = False
        self._env = None
        self._cv2_module = None

        self._load_env(env_name, env_params=env_params)

        self._stop_event = threading.Event()
        self._env_thread = threading.Thread(target=self._run_env)
        self._env_thread.daemon = True
        self._env_thread.start()

    def close(self):
        """Cleans up resources used by the program."""
        if self._env_thread is None:
            return
        self._stop_event.set()
        self._env_thread.join()
        self._stop_event = None
        self._env_thread = None

    def do_quit(self, _):
        """Quits the program."""
        self.close()
        return True

    def do_load(self, arg):
        """Loads an environment.

        A device name can be passed as a second argument to run on hardware.

        Example:
            # Loads a simulation environment:
            >>> load DClawTurnFixed-v0

            # Loads a hardware environment:
            >>> load DClawTurnFixed-v0 /dev/ttyUSB0
        """
        if not arg:
            return
        components = arg.split()
        if not components or len(components) > 2:
            print('An environment name, and optionally a device path must be '
                  'provided.')
            return
        env_name = components[0]
        env_params = components[1:] if len(components) > 1 else None
        self._load_env(env_name, env_params=env_params)

    def do_obs(self, _):
        """Prints out the current observation."""
        self._print_obs()

    def do_action(self, arg):
        """Sends an action to the environment.

        Example:
            >>> action [0] * 9
        """
        if not arg:
            return
        action = eval(arg)
        print('Sending action: {}'.format(action))

        action = np.array(action, dtype=np.float32)
        if not self._env.action_space.contains(action):
            print('Action in not in the action space: {}'.format(
                self._env.action_space))
            return

        with self._env_lock:
            self._do_random_actions = False
            self._action = action

    def do_set_state(self, arg):
        """Sets the state for a group."""
        if not arg:
            return
        state = eval(arg)
        print('Setting state to: {}'.format(state))
        state = np.array(state, dtype=np.float32)
        if not self._env.state_space.contains(state):
            print('State is not in the state space: {}'.format(
                self._env.state_space))
        with self._env_lock:
            self._env.set_state(state)

    def do_random(self, _):
        """Start doing random actions in the environment."""
        with self._env_lock:
            self._do_random_actions = True

    def do_reset(self, _):
        """Resets the environment.

        Example:
            >>> reset
        """
        with self._env_lock:
            self._do_random_actions = False
            self._action = None
            self._do_reset = True

    def do_engage(self, group_names):
        """Engages motors. This only affects hardware."""
        group_names = group_names.split()
        with self._env_lock:
            if self._env.robot.is_hardware:
                self._env.robot.set_motors_engaged(group_names, True)

    def do_disengage(self, group_names):
        """Disengages motors. This only affects hardware."""
        group_names = group_names.split()
        with self._env_lock:
            if self._env.robot.is_hardware:
                self._env.robot.set_motors_engaged(group_names, False)

    def emptyline(self):
        """Overrides behavior when an empty line is sent."""

    def precmd(self, line):
        """Overrides behavior when input is given."""
        line = line.strip()
        if any(line.startswith(prefix) for prefix in _ENV_COMMANDS):
            if self._env is None:
                print('No environment loaded.')
                return ''
        return line

    def _print_obs(self):
        """Prints the current observation."""
        assert self._env is not None
        with self._env_lock:
            obs = self._env.get_obs_dict()
        for key, value in obs.items():
            print('{}: {}'.format(key, value))

    def _load_env(self,
                  env_name: str,
                  device: Optional[str] = None,
                  env_params: Optional[Dict[str, Any]] = None):
        """Loads the given environment."""
        env_params = env_params or {}
        if device is not None:
            env_params['device_path'] = device
        if env_params:
            robel.set_env_params(env_name, env_params)

        with self._env_lock:
            self._load_env_name = env_name
            self._load_env_params = env_params

    def _run_env(self):
        """Runs a loop for the current environment."""
        step = 0
        while not self._stop_event.is_set():

            with self._env_lock:
                # Unload the current env and load the new one if given.
                if self._load_env_name is not None:
                    if self._env is not None:
                        self._env.close()
                    robel.set_env_params(self._load_env_name,
                                              self._load_env_params)
                    self._env = gym.make(self._load_env_name).unwrapped
                    self._env.reset()

                    self._load_env_name = None
                    self._load_env_params = None
                if self._env is None:
                    continue

            frame_start_time = time.time()

            with self._env_lock:
                if self._do_reset:
                    self._do_reset = False
                    self._env.reset()
                if self._do_random_actions:
                    self._action = self._env.action_space.sample()
                if self._action is not None:
                    self._env.step(self._action)
                else:
                    self._env.get_obs_dict()

                self._env.render()
                # self._display_image_obs(obs)

            sleep_duration = MIN_FRAME_TIME - (time.time() - frame_start_time)
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            step += 1

        with self._env_lock:
            if self._env is not None:
                self._env.close()
            self._env = None

    def _display_image_obs(self, obs):
        """Displays the given observation in a window."""
        if not isinstance(obs, collections.Mapping):
            return
        if self._cv2_module is None:
            try:
                import cv2
                self._cv2_module = cv2
            except ImportError:
                print('No cv2 module; not displaying images.')
                self._cv2_module = False
                return
        elif not self._cv2_module:
            return

        # Display a window for any image values.
        for key, value in obs.items():
            if not isinstance(value, np.ndarray):
                continue
            if value.ndim == 3 and value.shape[2] <= 4:
                # TODO(michaelahn): Consider data format and color space.
                self._cv2_module.imshow(key, value)
                self._cv2_module.waitKey(1)

    def __del__(self):
        self.close()


if __name__ == '__main__':
    env_id, params, _ = parse_env_args()
    logging.basicConfig(level=logging.INFO)
    repl = PlayShell(env_name=env_id, env_params=params)
    repl.cmdloop()
