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

"""Allows the user to interact with the OpenVR client."""

import cmd
import logging

from transforms3d.euler import euler2quat

from robel.components.tracking.virtual_reality.client import VrClient

INTRODUCTION = """Interactive shell for using the OpenVR client.

Type `help` or `?` to list commands.
"""


class VrDeviceShell(cmd.Cmd):
    """Implements a command-line interface for using the OpenVR client."""

    intro = INTRODUCTION
    prompt = '>>> '

    def __init__(self, client: VrClient):
        super().__init__()
        self.client = client

    def do_list(self, unused_arg):
        """Lists the available devices on the machine."""
        devices = self.client.discover_devices()
        if not devices:
            print('No devices found!')
            return
        for device in devices:
            print(device.get_summary())

    def do_pose(self, args):
        """Prints the pose for the given device."""
        names = args.split()
        devices = [self.client.get_device(name) for name in names]

        pose_batch = self.client.get_poses()
        for device in devices:
            state = pose_batch.get_state(device)
            print(device.get_summary())
            print('> Pos: [{:.3f} {:.3f} {:.3f}]'.format(*state.pos))
            print('> Rot: [{:.3f} {:.3f} {:.3f}]'.format(*state.rot_euler))

    def emptyline(self):
        """Overrides behavior when an empty line is sent."""


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    with VrClient() as vr_client:
        repl = VrDeviceShell(vr_client)
        repl.cmdloop()
