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

"""Tests for RobotGroupConfig."""

from typing import Any

from absl.testing import absltest
import numpy as np

from robel.components.robot.group_config import RobotGroupConfig
from robel.utils.testing.mock_sim_scene import MockSimScene


class RobotGroupConfigTest(absltest.TestCase):
    """Unit tests for RobotGroupConfig."""

    def test_qpos_indices(self):
        """Checks defaults when nq == nv == nu."""
        sim_scene = MockSimScene(nq=5, ctrl_range=(-1, 1))  # type: Any
        config = RobotGroupConfig(sim_scene, qpos_indices=range(5))
        result = np.array([0, 1, 2, 3, 4], dtype=int)

        np.testing.assert_array_equal(config.qpos_indices, result)
        np.testing.assert_array_equal(config.qvel_indices, result)
        np.testing.assert_array_equal(config.actuator_indices, result)

        np.testing.assert_array_equal(config.denormalize_center, np.zeros(5))
        np.testing.assert_array_equal(config.denormalize_range, np.ones(5))

    def test_qpos_out_of_range(self):
        """Ensures error when qpos indices are out of bounds."""
        sim_scene = MockSimScene(nq=3)  # type: Any
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qpos_indices=[3])
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qpos_indices=[-4])

    def test_qvel_indices(self):
        """Checks defaults when nq == nu != nv."""
        sim_scene = MockSimScene(nq=3, nv=5)  # type: Any
        config = RobotGroupConfig(
            sim_scene, qpos_indices=[-1], qvel_indices=[3, 4])

        np.testing.assert_array_equal(config.qpos_indices, [-1])
        np.testing.assert_array_equal(config.qvel_indices, [3, 4])
        np.testing.assert_array_equal(config.actuator_indices, [-1])

    def test_qvel_out_of_range(self):
        """Ensures error when qvel indices are out of bounds."""
        sim_scene = MockSimScene(nq=1, nv=3)  # type: Any
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qvel_indices=[3])
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qvel_indices=[-4])

    def test_qpos_range(self):
        """Checks presence of qpos_range when provided."""
        sim_scene = MockSimScene(nq=2)  # type: Any
        config = RobotGroupConfig(
            sim_scene, qpos_indices=[0, 1], qpos_range=[(0, 1)] * 2)

        np.testing.assert_array_equal(config.qpos_range,
                                      np.array([(0, 1), (0, 1)]))

    def test_qpos_invalid_range(self):
        """Ensures error when invalid qpos range is given."""
        sim_scene = MockSimScene(nq=2)  # type: Any
        with self.assertRaises(AssertionError):
            RobotGroupConfig(
                sim_scene, qpos_indices=[0, 1], qpos_range=[(0, 1)])
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qpos_indices=[0], qpos_range=[(1, 0)])

    def test_qvel_range(self):
        """Checks presence of qvel_range when provided."""
        sim_scene = MockSimScene(nq=2)  # type: Any
        config = RobotGroupConfig(
            sim_scene,
            qpos_indices=[0],
            qvel_indices=[0, 1],
            qvel_range=[(0, 1)] * 2)

        np.testing.assert_array_equal(config.qvel_range,
                                      np.array([(0, 1), (0, 1)]))

    def test_qvel_invalid_range(self):
        """Ensures error when invalid qvel range is given."""
        sim_scene = MockSimScene(nq=2, nv=3)  # type: Any
        with self.assertRaises(AssertionError):
            RobotGroupConfig(
                sim_scene, qvel_indices=[0, 2], qvel_range=[(-1, 1)] * 3)
        with self.assertRaises(AssertionError):
            RobotGroupConfig(sim_scene, qvel_indices=[0], qpos_range=[(-1, -2)])

    def test_actuator_range(self):
        """Checks presence of actuator_range when provided."""
        sim_scene = MockSimScene(nq=2, nu=3)  # type: Any
        config = RobotGroupConfig(
            sim_scene,
            qpos_indices=[0, 1],
            qpos_range=[(0, 1)] * 2,
            actuator_indices=[0, 1, 2],
            actuator_range=[(-1, 3)] * 3,
        )

        np.testing.assert_array_equal(config.actuator_range, [(-1, 3)] * 3)
        np.testing.assert_array_equal(config.denormalize_center, [1.] * 3)
        np.testing.assert_array_equal(config.denormalize_range, [2.] * 3)

    def test_actuator_range_default(self):
        """Checks that actuator_range uses the simulation range by default."""
        sim_scene = MockSimScene(nq=2)  # type: Any
        config = RobotGroupConfig(sim_scene, qpos_indices=[0, 1])

        np.testing.assert_array_equal(config.actuator_range, [(-1, 1)] * 2)


if __name__ == '__main__':
    absltest.main()
