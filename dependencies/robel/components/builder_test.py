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

"""Unit tests for ComponentBuilder."""

from absl.testing import absltest

from robel.components.builder import ComponentBuilder


class DummyBuilder(ComponentBuilder):
    """Mock component for testing ComponentBuilder."""

    def build(self, *args, **kwargs):
        """Builds the component."""


class ComponentBuilderTest(absltest.TestCase):
    """Unit test class for ComponentBuilder."""

    def test_add_group(self):
        """Tests adding a group."""
        builder = DummyBuilder()
        builder.add_group('test', a=1, b=2)
        self.assertDictEqual(builder.group_configs, {'test': dict(a=1, b=2)})
        self.assertListEqual(builder.group_names, ['test'])

    def test_add_multiple_groups(self):
        """Tests adding multiple groups."""
        builder = DummyBuilder()
        builder.add_group('test1', a=1, b=2)
        builder.add_group('test2', b=2, c=3)
        self.assertDictEqual(builder.group_configs, {
            'test1': dict(a=1, b=2),
            'test2': dict(b=2, c=3),
        })
        self.assertListEqual(builder.group_names, ['test1', 'test2'])

    def test_add_group_conflict(self):
        """Tests adding a duplicate group."""
        builder = DummyBuilder()
        builder.add_group('test')
        with self.assertRaises(ValueError):
            builder.add_group('test')
        self.assertListEqual(builder.group_names, ['test'])

    def test_update_group(self):
        """Tests updating an existing group."""
        builder = DummyBuilder()
        builder.add_group('test', a=1, b=2)
        builder.update_group('test', b=3, c=4)
        self.assertDictEqual(builder.group_configs,
                             {'test': dict(a=1, b=3, c=4)})

    def test_update_nonexistent_group(self):
        """Tests updating an non-existing group."""
        builder = DummyBuilder()
        builder.add_group('test', a=1, b=2)
        with self.assertRaises(ValueError):
            builder.update_group('nottest', b=3, c=4)
        self.assertDictEqual(builder.group_configs, {'test': dict(a=1, b=2)})


if __name__ == '__main__':
    absltest.main()
