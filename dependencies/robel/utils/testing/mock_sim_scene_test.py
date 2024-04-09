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

"""Unit tests for MockSimScene."""

from absl.testing import absltest
import numpy as np

from robel.utils.testing.mock_sim_scene import MockSimScene


class MockSimSceneTest(absltest.TestCase):
    """Tests MockSimScene."""

    def test_defaults(self):
        """Tests default initialization of the sim scene."""
        scene = MockSimScene(nq=2)

        # Ensure that properties exist.
        self.assertIsNotNone(scene.sim)
        self.assertIsNotNone(scene.model)
        self.assertIsNotNone(scene.data)
        self.assertIsNotNone(scene.step_duration)
        self.assertIsNotNone(scene.close)
        self.assertIsNotNone(scene.advance)

        np.testing.assert_array_equal(scene.model.actuator_ctrlrange, [(-1, 1),
                                                                       (-1, 1)])

        # Check that sizes are consistent.
        self.assertEqual(scene.model.nq, 2)
        self.assertEqual(scene.model.nv, 2)
        self.assertEqual(scene.model.nu, 2)
        self.assertEqual(scene.data.qpos.size, 2)
        self.assertEqual(scene.data.qvel.size, 2)
        self.assertEqual(scene.data.qacc.size, 2)
        self.assertEqual(scene.data.ctrl.size, 2)

    def test_explicit_init(self):
        """Tests initialization with explicit values."""
        scene = MockSimScene(
            nq=2,
            nv=3,
            nu=4,
            ctrl_range=(-2, 2),
            body_names=['test0'],
            geom_names=['test0', 'test1'],
            site_names=['test0', 'test1', 'test2'],
            cam_names=['cam0'],
            step_duration=0.5)

        self.assertEqual(scene.data.qpos.size, 2)
        self.assertEqual(scene.data.qvel.size, 3)
        self.assertEqual(scene.data.qacc.size, 3)
        self.assertEqual(scene.data.ctrl.size, 4)

        np.testing.assert_array_equal(scene.model.actuator_ctrlrange,
                                      [(-2, 2)] * 4)
        np.testing.assert_array_equal(scene.model.body_pos, np.zeros((1, 3)))
        np.testing.assert_array_equal(scene.model.geom_pos, np.zeros((2, 3)))
        np.testing.assert_array_equal(scene.model.site_pos, np.zeros((3, 3)))
        np.testing.assert_array_equal(scene.model.cam_pos, np.zeros((1, 3)))

        self.assertEqual(scene.model.body_name2id('test0'), 0)
        self.assertEqual(scene.model.geom_name2id('test1'), 1)
        self.assertEqual(scene.model.site_name2id('test2'), 2)
        self.assertEqual(scene.model.camera_name2id('cam0'), 0)

    def test_render_offscreen(self):
        """Tests mock rendering."""
        scene = MockSimScene(nq=1)
        image = scene.renderer.render_offscreen(16, 16)
        self.assertEqual(image.shape, (16, 16))


if __name__ == '__main__':
    absltest.main()
