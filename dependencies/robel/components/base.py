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

"""Base API for Components.

A Component provides a unified API between simulation and hardware.
"""

import abc
import logging
import sys
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from robel.simulation.sim_scene import SimScene

# Type definition for a group configuration.
GroupConfig = Any  # pylint: disable=invalid-name
GroupState = Any  # pylint: disable=invalid-name


class BaseComponent(abc.ABC):
    """Base class for all components."""
    def __init__(
        self,
        sim_scene: SimScene,
        groups: Dict[str, Dict],
        random_state: Optional[np.random.RandomState] = None,
    ):
        """Initializes a new component.

        Args:
            sim_scene: The simulation to control.
            groups: Group configurations for reading/writing state.
            random_state: A random state to use for generating noise.
        """
        self.sim_scene = sim_scene
        self.random_state = random_state

        if self.random_state is None:
            logging.info(
                'Random state not given; observation noise will be ignored')

        # Process all groups.
        self.groups = {}
        for group_name, group_config in groups.items():
            try:
                config = self._process_group(**group_config)
            except Exception as e:
                raise type(e)('[{}] Error parsing group "{}": {}'.format(
                    self.__class__.__name__,
                    group_name,
                    str(e),
                )).with_traceback(sys.exc_info()[2])
            self.groups[group_name] = config

    @property
    def is_hardware(self) -> bool:
        """Returns True if this is a hardware component."""
        return False

    def close(self):
        """Cleans up any resources used by the component."""

    def get_state(self, groups: Union[str, Sequence[str]],
                  **kwargs) -> Union[GroupState, Sequence[GroupState]]:
        """Returns the state of the given groups.

        Args:
            groups: Either a single group name or a list of group names of the
                groups to retrieve the state of.

        Returns:
            If `groups` is a string, returns a single state object. Otherwise,
            returns a list of state objects.
        """
        if isinstance(groups, str):
            states = self._get_group_states([self.get_config(groups)], **kwargs)
        else:
            states = self._get_group_states(
                [self.get_config(name) for name in groups], **kwargs)

        if isinstance(groups, str):
            return states[0]
        return states

    def get_config(self, group_name: str) -> GroupConfig:
        """Returns the configuration for a group."""
        if group_name not in self.groups:
            raise ValueError(
                'Group "{}" is not in the configured groups: {}'.format(
                    group_name, sorted(self.groups.keys())))
        return self.groups[group_name]

    @abc.abstractmethod
    def _process_group(self, **config_kwargs) -> GroupConfig:
        """Processes the configuration for a group.

        This should be overridden by subclasses to define and validate the group
        configuration.

        Args:
            **config_kwargs: Keyword arguments from the group configuration.

        Returns:
            An object that defines the group.
            e.g. A class that stores the group parameters.
        """

    @abc.abstractmethod
    def _get_group_states(self, configs: Sequence[GroupConfig],
                          **kwargs) -> Sequence[GroupState]:
        """Returns the states for the given group configurations."""
