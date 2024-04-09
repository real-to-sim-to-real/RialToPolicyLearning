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

"""Builder-specific logic for creating TrackerComponents."""

import enum
from typing import Any, Dict, Optional, Union

from robel.components.builder import ComponentBuilder
from robel.components.tracking.tracker import TrackerComponent
from robel.components.tracking.vr_tracker import VrTrackerComponent
from robel.components.tracking.phasespace_tracker import PhaseSpaceTrackerComponent


class TrackerType(enum.Enum):
    """The type of the tracker."""
    SIMULATED = 0
    OPENVR = 1
    PHASESPACE = 2


class TrackerComponentBuilder(ComponentBuilder):
    """Builds a RobotComponent."""

    def __init__(self):
        super().__init__()
        self._hardware_tracker_ids = set()
        self._tracker_type = None
        self._tracker_kwargs = {}

    def build(self, *args, **kwargs):
        """Builds the component."""
        kwargs = {**kwargs, **self._tracker_kwargs}
        tracker_type = self._tracker_type
        if tracker_type is None:
            # Default to OpenVR if tracker IDs were added.
            if self._hardware_tracker_ids:
                tracker_type = TrackerType.OPENVR
            else:
                tracker_type = TrackerType.SIMULATED

        if tracker_type == TrackerType.OPENVR:
            return VrTrackerComponent(
                *args, groups=self.group_configs, **kwargs)
        elif tracker_type == TrackerType.PHASESPACE:
            return PhaseSpaceTrackerComponent(
                *args, groups=self.group_configs, **kwargs)
        elif tracker_type == TrackerType.SIMULATED:
            return TrackerComponent(
                *args, groups=self.group_configs, **kwargs)
        else:
            raise NotImplementedError(self._tracker_type)

    def set_tracker_type(self,
                         tracker_type: Union[TrackerType, str],
                         **kwargs):
        """Sets the tracker type."""
        if isinstance(tracker_type, str):
            tracker_type = TrackerType[tracker_type.upper()]
        self._tracker_type = tracker_type
        self._tracker_kwargs = kwargs

    def set_hardware_tracker_id(self,
                                group_name: str,
                                tracker_id: Union[str, int]):
        """Sets the hardware tracker ID for the given group.

        Args:
            group_name: The group to set the tracker ID for.
            tracker_id: Either the device serial number string, or the device
                index of the tracking device.
        """
        self._check_group_exists(group_name)
        self._hardware_tracker_ids.add(tracker_id)
        self.group_configs[group_name]['device_identifier'] = tracker_id

    def add_tracker_group(
            self,
            group_name: str,
            hardware_tracker_id: Optional[Union[str, int]],
            sim_params: Optional[Dict[str, Any]] = None,
            hardware_params: Optional[Dict[str, Any]] = None,
            mimic_sim: bool = True,
            mimic_xy_only: bool = False,
    ):
        """Convenience method for adding a tracking group.

        Args:
            group_name: The group name to create.
            hardware_tracker_id: Either the device serial number string, or
                the device index of the hardware tracking device.
            sim_params: The group parameters for simulation tracking.
            hardware_params: The group parameters for hardware tracking.
            mimic_sim: If True, adds parameters so that simulation mimics the
                hardware.
            mimic_xy_only: If True, adds parameters so that only the XY plane
                movement is mimicked to the simulation.
        """
        self.add_group(group_name, **sim_params)
        if hardware_tracker_id is not None:
            self.set_hardware_tracker_id(group_name, hardware_tracker_id)

            hardware_params = hardware_params or {}
            if mimic_sim:
                hardware_params['mimic_in_sim'] = True
            if mimic_xy_only:
                hardware_params.update({
                    'mimic_ignore_z_axis': True,
                    'mimic_ignore_rotation': True,
                })
            if hardware_params:
                self.update_group(group_name, **hardware_params)
