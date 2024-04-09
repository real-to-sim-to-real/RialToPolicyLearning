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

"""Builder-specific logic for creating RobotComponents."""

from robel.components.builder import ComponentBuilder
from robel.components.robot.dynamixel_utils import CalibrationMap


class RobotComponentBuilder(ComponentBuilder):
    """Builds a RobotComponent."""

    def __init__(self):
        super().__init__()
        self._dxl_device_path = None
        self._calibration_map = None

    def build(self, *args, **kwargs):
        """Builds the component."""
        if self._dxl_device_path:
            if self._calibration_map is not None:
                self._calibration_map.update_group_configs(self.group_configs)
            from robel.components.robot.dynamixel_robot import (
                DynamixelRobotComponent)
            return DynamixelRobotComponent(
                *args,
                groups=self.group_configs,
                device_path=self._dxl_device_path,
                **kwargs)
        from robel.components.robot.robot import RobotComponent
        return RobotComponent(*args, groups=self.group_configs, **kwargs)

    def set_dynamixel_device_path(self, device_path: str):
        """Sets the device path for a hardware robot component.

        If set, the builder will build a DynamixelRobotComponent.

        Args:
            device_path: The device path to the Dynamixel device.
        """
        self._dxl_device_path = device_path

    def set_hardware_calibration_map(self, calibration_map: CalibrationMap):
        """Sets the calibration map for hardware."""
        self._calibration_map = calibration_map
