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

"""Tests for resources."""

from absl.testing import absltest

from robel.utils.resources import AssetBundle


class DummyResources:
    """Dummy cache of resources."""

    def __init__(self, assets):
        self.assets = assets

    def get_resource(self, path: str):
        return self.assets[path]


class TestAssetBundle(absltest.TestCase):
    """Unit tests for configurable."""

    def test_add_mujoco(self):
        """Tests adding a MuJoCo file."""
        resources = DummyResources({
            'a/b/main.xml': '<mujoco><include file="../child1.xml"/></mujoco>',
            'a/child1.xml': """
                <mujoco>
                    <compiler meshdir="c"/>
                    <asset>
                        <mesh name="a1" file="hello.stl"/>
                        <mesh name="a2" file="world.stl"/>
                    </asset>
                </mujoco>
            """,
            'a/c/hello.stl': 'Hello!',
            'a/c/world.stl': 'World!',
        })
        bundle = AssetBundle(
            dest_path='test', dry_run=True, resource_fn=resources.get_resource)

        transformed_path = bundle.add_mujoco('a/b/main.xml')
        self.assertEqual(transformed_path, 'test/a/b/main.xml')
        self.assertDictEqual(
            bundle.copied_paths, {
                'a/b/main.xml': 'test/a/b/main.xml',
                'a/child1.xml': 'test/a/child1.xml',
                'a/c/hello.stl': 'test/a/c/hello.stl',
                'a/c/world.stl': 'test/a/c/world.stl',
            })


if __name__ == '__main__':
    absltest.main()
