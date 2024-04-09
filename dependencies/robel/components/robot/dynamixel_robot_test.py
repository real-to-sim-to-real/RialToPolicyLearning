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

"""Unit tests for RobotComponent and RobotGroupConfig."""

from typing import Any, Sequence

from absl.testing import absltest
from absl.testing.absltest import mock
import numpy as np

from robel.components.robot.dynamixel_robot import (
    DynamixelRobotComponent, RobotState)
from robel.utils.testing.mock_sim_scene import MockSimScene
from robel.utils.testing.mock_time import patch_time


class MockDynamixelClient:
    """Mock class for dynamixel_py."""

    def __init__(self, all_motor_ids: Sequence[int], **unused_kwargs):
        self.motor_ids = np.array(sorted(all_motor_ids), dtype=int)
        self.qpos = np.zeros_like(self.motor_ids, dtype=np.float32)
        self.qvel = np.zeros_like(self.motor_ids, dtype=np.float32)
        self.current = np.zeros_like(self.motor_ids, dtype=np.float32)
        self.enabled = np.zeros_like(self.motor_ids, dtype=bool)
        self.is_connected = True

    def set_torque_enabled(self, motor_ids: Sequence[int], enabled: bool):
        assert self.is_connected
        indices = np.searchsorted(self.motor_ids, motor_ids)
        self.enabled[indices] = enabled

    def read_pos_vel_cur(self):
        assert self.is_connected
        return self.qpos, self.qvel, self.current

    def write_desired_pos(self, motor_ids: Sequence[int],
                          qpos: Sequence[float]):
        assert self.is_connected
        indices = np.searchsorted(self.motor_ids, motor_ids)
        self.qpos[indices] = qpos


def patch_dynamixel(test_fn):
    """Decorator to patch dynamixel_py with a mock."""

    def patched_fn(self):
        DynamixelRobotComponent.DEVICE_CLIENTS.clear()
        with mock.patch(
                'robel.components.robot.dynamixel_robot.DynamixelClient',
                new=MockDynamixelClient):
            test_fn(self)

    return patched_fn


class RobotComponentTest(absltest.TestCase):
    """Unit test class for RobotComponent."""

    @patch_dynamixel
    def test_get_state(self):
        """Tests querying the state of multiple groups."""
        sim_scene = MockSimScene(nq=10)  # type: Any
        robot = DynamixelRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 3, 5],
                    'motor_ids': [10, 20, 12, 21],
                    'calib_scale': [0.5] * 4,
                    'calib_offset': [1] * 4,
                },
                'b': {
                    'qpos_indices': [2, 6],
                },
                'c': {
                    'motor_ids': [22, 24],
                },
            },
            device_path='test')
        dxl = DynamixelRobotComponent.DEVICE_CLIENTS['test']
        dxl.write_desired_pos([10, 12, 20, 21, 22, 24], [1, 2, 3, 4, 5, 6])

        a_state, b_state, c_state = robot.get_state(['a', 'b', 'c'])
        np.testing.assert_allclose(a_state.qpos, [1.5, 2.5, 2., 3.])
        np.testing.assert_allclose(a_state.qvel, [0., 0., 0., 0.])
        self.assertIsNone(b_state.qpos)
        self.assertIsNone(b_state.qvel)
        np.testing.assert_allclose(c_state.qpos, [5., 6.])
        np.testing.assert_allclose(c_state.qvel, [0., 0.])

        np.testing.assert_allclose(sim_scene.data.qpos,
                                   [1.5, 2.5, 0, 2., 0, 3., 0, 0, 0, 0])

    @patch_dynamixel
    def test_step(self):
        """Tests stepping with an action for multiple groups."""
        sim_scene = MockSimScene(nq=10, ctrl_range=[-1, 1])  # type: Any
        robot = DynamixelRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 2],
                    'motor_ids': [10, 11, 12],
                    'calib_offset': [1] * 3,
                    'calib_scale': [-1] * 3,
                    'qpos_range': [(-0.5, 0.5)] * 3,
                },
                'b': {
                    'qpos_indices': [2, 3],
                },
            },
            device_path='test')
        with patch_time('robel.components.robot.hardware_robot.time'):
            robot.step({
                'a': np.array([.2, .4, .6]),
                'b': np.array([.1, .3]),
            })
        dxl = DynamixelRobotComponent.DEVICE_CLIENTS['test']  # type: Any

        np.testing.assert_allclose(dxl.qpos, [.9, .8, .7])

    @patch_dynamixel
    def test_set_state(self):
        """Tests stepping with an action for multiple groups."""
        sim_scene = MockSimScene(nq=10)  # type: Any
        robot = DynamixelRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 2],
                    'motor_ids': [10, 11, 12],
                    'calib_offset': [-1] * 3,
                    'qpos_range': [(-2.5, 2.5)] * 3,
                },
            },
            device_path='test')
        with patch_time('robel.components.robot.hardware_robot.time'):
            robot.set_state({
                'a': RobotState(qpos=np.array([1, 2, 3])),
            })
        dxl = DynamixelRobotComponent.DEVICE_CLIENTS['test']  # type: Any

        np.testing.assert_allclose(dxl.qpos, [2, 3, 3.5])

    @patch_dynamixel
    def test_engage_motors(self):
        """Tests engaging/disengaging subsets of motors."""
        sim_scene = MockSimScene(nq=10)  # type: Any
        robot = DynamixelRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'motor_ids': [10, 11, 12],
                },
                'b': {
                    'motor_ids': [13, 14, 15],
                },
                'c': {
                    'motor_ids': [12, 15],
                }
            },
            device_path='test')
        dxl = DynamixelRobotComponent.DEVICE_CLIENTS['test']  # type: Any
        np.testing.assert_array_equal(dxl.enabled, [False] * 6)

        robot.set_motors_engaged('a', True)
        np.testing.assert_array_equal(dxl.enabled, [True] * 3 + [False] * 3)

        robot.set_motors_engaged('c', False)
        np.testing.assert_array_equal(dxl.enabled,
                                      [True, True, False, False, False, False])

        robot.set_motors_engaged(['a', 'b'], True)
        np.testing.assert_array_equal(dxl.enabled, [True] * 6)


if __name__ == '__main__':
    absltest.main()
