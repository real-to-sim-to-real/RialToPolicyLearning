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

"""Shared logic for a component builder.
"""

import abc
from typing import List

from robel.components.base import BaseComponent


class ComponentBuilder(metaclass=abc.ABCMeta):
    """Base class for a component configuration.

    This wraps a dictionary of parameters that is used to initialize a
    Component.
    """

    def __init__(self):
        self.group_configs = {}

    @property
    def group_names(self) -> List[str]:
        """Returns the sorted list of current group names."""
        return sorted(self.group_configs.keys())

    @abc.abstractmethod
    def build(self, *args, **kwargs) -> BaseComponent:
        """Builds the component."""

    def add_group(self, group_name: str, **group_kwargs):
        """Adds a group configuration.

        Args:
            group_name: The name of the group.
            **group_kwargs: Key-value pairs to configure the group with.
        """
        if group_name in self.group_configs:
            raise ValueError(
                'Group with name `{}` already exists in component config.'
                .format(group_name))
        self.group_configs[group_name] = group_kwargs

    def update_group(self, group_name: str, **group_kwargs):
        """Updates a group configuration.

        Args:
            group_name: The name of the group.
            **group_kwargs: Key-value pairs to configure the group with.
        """
        self._check_group_exists(group_name)
        self.group_configs[group_name].update(group_kwargs)

    def _check_group_exists(self, name: str):
        """Raises an error if a group with the given name doesn't exist."""
        if name not in self.group_configs:
            raise ValueError(
                ('No group with name "{}" was added to the builder. Currently '
                 'added groups: {}').format(name, self.group_names))
