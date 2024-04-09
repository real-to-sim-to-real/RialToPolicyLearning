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

"""Device-related logic from OpenVR devices."""

import openvr


class VrDevice:
    """Represents an OpenVR device."""

    def __init__(self, vr_system, index: int):
        """Initializes a new VrDevice with the given device index."""
        self._vr_system = vr_system
        self._index = index
        self._value_cache = {}

    @property
    def index(self) -> int:
        """Returns the device index."""
        return self._index

    def is_connected(self) -> bool:
        """Returns whether the device is connected."""
        return self._vr_system.isTrackedDeviceConnected(self._index)

    def get_serial(self) -> str:
        """Returns the serial number of the device."""
        return self._get_string(openvr.Prop_SerialNumber_String)

    def get_model(self) -> str:
        """Returns the model number of the device."""
        return self._get_string(openvr.Prop_ModelNumber_String)

    def get_model_name(self) -> str:
        """Returns the model name of the device."""
        return self._get_string(openvr.Prop_RenderModelName_String)

    def get_summary(self) -> str:
        """Returns a summary of information about the device."""
        connected = self.is_connected()
        info = '[{} - {}]'.format(self._index,
                                  'Connected' if connected else 'Disconnected')
        if connected:
            info += ' {} ({})'.format(self.get_model_name(), self.get_serial())
        return info

    def _get_string(self, prop_type) -> str:
        """Returns a string property of the device."""
        if prop_type in self._value_cache:
            return self._value_cache[prop_type]
        value = self._vr_system.getStringTrackedDeviceProperty(
            self._index, prop_type).decode('utf-8')
        self._value_cache[prop_type] = value
        return value

    def _get_bool(self, prop_type) -> bool:
        """Returns a boolean property of the device."""
        if prop_type in self._value_cache:
            return self._value_cache[prop_type]
        value = self._vr_system.getBoolTrackedDeviceProperty(
            self._index, prop_type)[0]
        self._value_cache[prop_type] = value
        return value

    def _get_float(self, prop_type) -> float:
        """Returns a float property of the device."""
        if prop_type in self._value_cache:
            return self._value_cache[prop_type]
        value = self._vr_system.getFloatTrackedDeviceProperty(
            self._index, prop_type)[0]
        self._value_cache[prop_type] = value
        return value
