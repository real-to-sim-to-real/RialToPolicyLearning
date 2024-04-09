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

import time

import numpy as np
from transforms3d.quaternions import quat2mat

from robel.components.tracking.hardware_tracker import (
    DeviceId, HardwareTrackerComponent, HardwareTrackerGroupConfig,
    TrackerState)

# Seconds to sleep to ensure PhaseSpace can start getting data.
PHASESPACE_INIT_TIME = 4


class PhaseSpaceTrackerGroupConfig(HardwareTrackerGroupConfig):
    """Stores group configuration for a PhaseSpaceTrackerComponent."""


class PhaseSpaceTrackerComponent(HardwareTrackerComponent):
    """Component for reading tracking data from PhaseSpace."""

    # Cached client that is shared for the application lifetime.
    _PS_CLIENT = None

    def __init__(self, *args, server_address: str, **kwargs):
        """Initializes a VrTrackerComponent."""
        super().__init__(*args, **kwargs)
        if self._PS_CLIENT is None:
            import phasespace
            print('Connecting to PhaseSpace at: {}'.format(server_address))
            self._PS_CLIENT = phasespace.PhaseSpaceClient(server_address)
            print('Connected! Waiting for initialization...')
            time.sleep(PHASESPACE_INIT_TIME)
        self._position_scale = 1e-3
        self._poses = None
        self._state_cache = {}

    def _process_group(self, **config_kwargs):
        """Processes the configuration for a group."""
        return PhaseSpaceTrackerGroupConfig(self.sim_scene, **config_kwargs)

    def _refresh_poses(self):
        """Refreshes the pose state."""
        self._poses = self._PS_CLIENT.get_state()

    def _get_raw_tracker_state(self, device_id: DeviceId):
        """Returns the tracker state."""
        try:
            rigid_data = self._poses.get_rigid(device_id)
            state = TrackerState(
                pos=rigid_data.position * self._position_scale,
                rot=quat2mat(rigid_data.rotation),
                vel=np.zeros(3),
                angular_vel=np.zeros(3))
            self._state_cache[device_id] = state
        except:
            if device_id not in self._state_cache:
                raise
            state = self._state_cache[device_id]
        return state
