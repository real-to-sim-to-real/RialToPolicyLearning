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

"""Implementation of TrackerComponent using OpenVR to track devices."""

import numpy as np

from robel.components.tracking.hardware_tracker import (
    DeviceId, HardwareTrackerComponent, HardwareTrackerGroupConfig,
    TrackerState)


class VrTrackerGroupConfig(HardwareTrackerGroupConfig):
    """Stores group configuration for a VrTrackerComponent."""


class VrTrackerComponent(HardwareTrackerComponent):
    """Component for reading tracking data from a HTC Vive."""

    # Cached VR client that is shared for the application lifetime.
    _VR_CLIENT = None

    # Transform from OpenVR space (y upwards) to simulation space (z upwards).
    GLOBAL_TRANSFORM = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]],
                                dtype=np.float32)

    def __init__(self, *args, **kwargs):
        """Initializes a VrTrackerComponent."""
        super().__init__(*args, **kwargs)
        self._coord_system.set_coordinate_transform(self.GLOBAL_TRANSFORM)
        self._poses = None

        if self._VR_CLIENT is None:
            from robel.components.tracking.virtual_reality.client import (
                VrClient)
            self._VR_CLIENT = VrClient()

        # Check that all devices exist.
        for name, group in self.groups.items():
            if group.device_identifier is None:
                continue
            device = self._VR_CLIENT.get_device(group.device_identifier)
            print('Assigning group "{}" with device: {}'.format(
                name, device.get_summary()))

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return VrTrackerGroupConfig(self.sim_scene, **config_kwargs)

    def _refresh_poses(self):
        """Refreshes the pose state."""
        self._poses = self._VR_CLIENT.get_poses()

    def _get_raw_tracker_state(self, device_id: DeviceId) -> TrackerState:
        """Returns the tracker state."""
        device = self._VR_CLIENT.get_device(device_id)
        return self._poses.get_state(device)
