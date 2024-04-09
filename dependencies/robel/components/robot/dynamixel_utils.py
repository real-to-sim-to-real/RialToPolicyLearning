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

"""Helper classes and methods for working with Dynamixel robots."""

from typing import Any, Dict, Iterable, Tuple


class CalibrationMap:
    """Mapping of motor ID to (scale, offset) tuples."""

    def __init__(self, mapping: Dict[int, Tuple[float, float]]):
        """Initializes a new calibration mapping."""
        self.mapping = mapping

    def get_parameters(self, motor_ids: Iterable[int]) -> Dict[str, Any]:
        """Returns a dictionary of calibration parameters.

        Args:
            motor_ids: The motor IDs to get calibration parameters for.
        """
        return {
            'calib_scale': [self.mapping[i][0] for i in motor_ids],
            'calib_offset': [self.mapping[i][1] for i in motor_ids],
        }

    def make_group_config(self, motor_ids: Iterable[int],
                          **group_kwargs) -> Dict[str, Any]:
        """Creates a Dynamixel group configuration for the given motor IDs.

        Args:
            motor_ids: The motor IDs to create a group for.
            **group_kwargs: Additional configuration to set for the group.
        """
        return {
            'motor_ids': motor_ids,
            **self.get_parameters(motor_ids),
            **group_kwargs
        }

    def update_group_configs(self, configs: Dict[str, Any]):
        """Updates the calibration values for groups in the configuration.

        Args:
            config: The component configuration to update.
            *group_names: One or more group names to update.
        """
        for group_config in configs.values():
            if 'motor_ids' not in group_config:
                continue
            group_config.update(self.get_parameters(group_config['motor_ids']))
