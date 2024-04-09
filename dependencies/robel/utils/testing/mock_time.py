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

"""Utilities to mock time-related methods."""

from absl.testing.absltest import mock


class MockTime:
    """Class to mock the functionality of the time module."""

    def __init__(self, initial_time: float = 0.0):
        self._time = initial_time

    def time(self) -> float:
        return self._time

    def sleep(self, duration: float):
        self._time += duration


def patch_time(module_path: str, **kwargs):
    return mock.patch(module_path, MockTime(**kwargs))
