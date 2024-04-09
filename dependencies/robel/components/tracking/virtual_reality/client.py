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

"""Client to communicate with a VR device using OpenVR.

Example usage:
>>> client = VrClient()
>>> client.set_devices({'tracker': 1})
"""

from typing import List, Union

import openvr

from robel.components.tracking.virtual_reality.device import VrDevice
from robel.components.tracking.virtual_reality.poses import VrPoseBatch


class VrClient:
    """Communicates with a VR device."""

    def __init__(self):
        self._vr_system = None
        self._devices = []
        self._device_serial_lookup = {}
        self._device_index_lookup = {}
        self._last_pose_batch = None
        self._plot = None

        # Attempt to start OpenVR.
        if not openvr.isRuntimeInstalled():
            raise OSError('OpenVR runtime not installed.')

        self._vr_system = openvr.init(openvr.VRApplication_Other)

    def close(self):
        """Cleans up any resources used by the client."""
        if self._vr_system is not None:
            openvr.shutdown()
            self._vr_system = None

    def get_device(self, identifier: Union[int, str]) -> VrDevice:
        """Returns the device with the given name."""
        identifier = str(identifier)
        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        if identifier in self._device_serial_lookup:
            return self._device_serial_lookup[identifier]

        self.discover_devices()
        if (identifier not in self._device_index_lookup
                and identifier not in self._device_serial_lookup):
            raise ValueError(
                'Could not find device with name or index: {} (Available: {})'
                .format(identifier, sorted(self._device_serial_lookup.keys())))

        if identifier in self._device_index_lookup:
            return self._device_index_lookup[identifier]
        return self._device_serial_lookup[identifier]

    def discover_devices(self) -> List[VrDevice]:
        """Returns and caches all connected devices."""
        self._device_index_lookup.clear()
        self._device_serial_lookup.clear()
        devices = []
        for device_index in range(openvr.k_unMaxTrackedDeviceCount):
            device = VrDevice(self._vr_system, device_index)
            if not device.is_connected():
                continue
            devices.append(device)
            self._device_index_lookup[str(device.index)] = device
            self._device_serial_lookup[device.get_serial()] = device
        self._devices = devices
        return devices

    def get_poses(self, time_from_now: float = 0.0,
                  update_plot: bool = True) -> VrPoseBatch:
        """Returns a batch of poses that can be queried per device.

        Args:
            time_from_now: The seconds into the future to read poses.
            update_plot: If True, updates an existing plot.
        """
        pose_batch = VrPoseBatch(self._vr_system, time_from_now)
        self._last_pose_batch = pose_batch
        if update_plot and self._plot and self._plot.is_open:
            self._plot.refresh()
        return pose_batch

    def __enter__(self):
        """Enables use as a context manager."""
        return self

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.close()

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.close()
