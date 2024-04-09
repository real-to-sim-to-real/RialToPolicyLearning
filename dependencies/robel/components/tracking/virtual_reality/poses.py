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

"""Pose-related logic for OpenVR devices."""

import numpy as np
import openvr

from robel.components.tracking.tracker import TrackerState
from robel.components.tracking.virtual_reality.device import VrDevice


class VrPoseBatch:
    """Represents a batch of poses calculated by the OpenVR system."""

    def __init__(self, vr_system, time_from_now: float = 0.0):
        """Initializes a new pose batch."""
        self._vr_system = vr_system
        # Query poses for all devices.
        self.poses = self._vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, time_from_now,
            openvr.k_unMaxTrackedDeviceCount)

    def get_state(self, device: VrDevice) -> TrackerState:
        """Returns the tracking state for the given device."""
        if not self.poses[device.index].bPoseIsValid:
            state = TrackerState()
        else:
            vr_pose = np.ctypeslib.as_array(
                self.poses[device.index].mDeviceToAbsoluteTracking[:],
                shape=(3, 4))
            # Check that the pose is valid.
            # If all of the translations are 0, get from the cache.
            assert vr_pose.shape == (3, 4)
            state = TrackerState(pos=vr_pose[:, 3], rot=vr_pose[:, :3])
        return state
