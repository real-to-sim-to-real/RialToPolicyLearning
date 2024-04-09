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

from typing import Any

from absl.testing import absltest
import numpy as np

from robel.components.robot.robot import ControlMode, RobotComponent
from robel.utils.testing.mock_sim_scene import MockSimScene


class RobotComponentTest(absltest.TestCase):
    """Unit test class for RobotComponent."""

    def test_get_state(self):
        """Tests querying the state of multiple groups."""
        sim_scene = MockSimScene(nq=10)  # type: Any
        sim_scene.data.qpos[:] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        sim_scene.data.qvel[:] = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        robot = RobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 3, 5],
                },
                'b': {
                    'qpos_indices': [2, 6],
                    'qvel_indices': [7, 8, 9],
                },
            })
        a_state, b_state = robot.get_state(['a', 'b'])
        np.testing.assert_array_equal(a_state.qpos, [1, 2, 4, 6])
        np.testing.assert_array_equal(a_state.qvel, [11, 12, 14, 16])
        np.testing.assert_array_equal(b_state.qpos, [3, 7])
        np.testing.assert_array_equal(b_state.qvel, [18, 19, 20])

    def test_step(self):
        """Tests stepping with an action for multiple groups."""
        sim_scene = MockSimScene(nq=10, ctrl_range=[-1, 1])  # type: Any
        robot = RobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 3, 5],
                },
                'b': {
                    'actuator_indices': [7, 8, 9],
                },
            })
        robot.step({
            'a': np.array([.2, .4, .6, .8]),
            'b': np.array([.1, .3, .5])
        })
        np.testing.assert_allclose(sim_scene.data.ctrl,
                                   [.2, .4, 0, .6, 0, .8, 0, .1, .3, .5])

        self.assertEqual(sim_scene.advance.call_count, 1)

    def test_step_denormalize(self):
        """Tests denormalizing the actions to the sim control range."""
        sim_scene = MockSimScene(nq=5, ctrl_range=[0, 10])  # type: Any
        robot = RobotComponent(
            sim_scene, groups={'a': {
                'qpos_indices': [0, 1, 2, 3, 4],
            }})
        robot.step({
            'a': np.array([-1, 1, -0.5, 0.5, 0]),
        })
        np.testing.assert_allclose(sim_scene.data.ctrl, [0, 10, 2.5, 7.5, 5])

    def test_step_position_control_bounds(self):
        """Tests action clamping when doing position control."""
        sim_scene = MockSimScene(nq=5, ctrl_range=[-1, 1])  # type: Any
        sim_scene.data.qpos[:] = [-0.4, -0.2, 0, 0.2, 0.4]
        robot = RobotComponent(
            sim_scene,
            groups={
                'a': {
                    'qpos_indices': [0, 1, 2, 3, 4],
                    'qpos_range': [(-0.5, 0.5)] * 5,
                    'qvel_range': [(-0.2, 0.2)] * 5,
                }
            })
        robot.step({'a': np.array([-1, -1, 0.2, 1, 1])})
        np.testing.assert_allclose(sim_scene.data.ctrl,
                                   [-0.5, -0.4, 0.1, 0.4, 0.5])

    def test_step_velocity_control_bounds(self):
        """Tests action clamping when doing velocity control."""
        sim_scene = MockSimScene(nq=3, ctrl_range=[-10, 10])  # type: Any
        robot = RobotComponent(
            sim_scene,
            groups={
                'a': {
                    'control_mode': ControlMode.JOINT_VELOCITY,
                    'qpos_indices': [0, 1, 2],
                    'qvel_range': [(-2, 2), (-1, 5), (-7, -4)],
                }
            })
        robot.step({'a': np.array([-0.5, 1, -1])})
        np.testing.assert_allclose(sim_scene.data.ctrl, [-1, 5, -7])


if __name__ == '__main__':
    absltest.main()
