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

"""Unit tests for BaseComponent."""

from typing import Any

from absl.testing import absltest

from robel.components.base import BaseComponent
from robel.utils.testing.mock_sim_scene import MockSimScene


class DummyComponent(BaseComponent):
    """Mock component for testing BaseComponent."""

    def __init__(self, **kwargs):
        sim_scene = MockSimScene(nq=1)  # type: Any
        super().__init__(sim_scene=sim_scene, **kwargs)

    def _process_group(self, **config_kwargs):
        return {}

    def _get_group_states(self, configs):
        return [0 for group in configs]


class BaseComponentTest(absltest.TestCase):
    """Unit test class for BaseComponent."""

    def test_get_state(self):
        """Tests retrieving state from a single group."""
        component = DummyComponent(groups={'foo': {}})
        state = component.get_state('foo')
        self.assertEqual(state, 0)

    def test_get_states(self):
        """Tests retrieving state from multiple groups."""
        component = DummyComponent(groups={'foo': {}, 'bar': {}})
        foo_state, bar_state = component.get_state(['foo', 'bar'])
        self.assertEqual(foo_state, 0)
        self.assertEqual(bar_state, 0)


if __name__ == '__main__':
    absltest.main()
