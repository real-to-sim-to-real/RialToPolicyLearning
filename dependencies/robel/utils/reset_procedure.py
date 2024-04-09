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

"""Manages resetting functionality."""

import abc

from robel.components.builder import ComponentBuilder


class ResetProcedure(metaclass=abc.ABCMeta):
    """Implements a reset procedure for a robot."""

    def __init__(self):
        """Creates a new reset procedure."""

    @abc.abstractmethod
    def configure_reset_groups(self, builder: ComponentBuilder):
        """Configures the component groups needed for reset."""

    @abc.abstractmethod
    def reset(self, **kwargs):
        """Performs the reset procedure."""

    def finish(self):
        """Called when the reset is complete."""


class ManualResetProcedure(ResetProcedure):
    """Reset procedure that waits for the user to press enter."""

    def configure_reset_groups(self, builder: ComponentBuilder):
        """Configures the component groups needed for reset."""

    def reset(self, **kwargs):
        """Performs the reset procedure."""

    def finish(self):
        """Called when the reset is complete."""
        input('Press Enter to start the episode...')
