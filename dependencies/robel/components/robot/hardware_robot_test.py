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

from robel.components.robot.hardware_robot import (HardwareRobotComponent,
                                                        RobotState)
from robel.utils.testing.mock_sim_scene import MockSimScene
from robel.utils.testing.mock_time import patch_time


class DummyHardwareRobotComponent(HardwareRobotComponent):
    """Test implementation of HardwareRobotComponent."""

    def calibrate_state(self, state: RobotState, group_name: str):
        self._calibrate_state(state, self.get_config(group_name))

    def decalibrate_qpos(self, qpos: np.ndarray, group_name: str):
        return self._decalibrate_qpos(qpos, self.get_config(group_name))

    def do_timestep(self):
        self._synchronize_timestep()


class HardwareRobotComponentTest(absltest.TestCase):
    """Unit test class for HardwareRobotComponent."""

    def test_calibrate_state(self):
        """Converts a state to component space."""
        sim_scene = MockSimScene(nq=1)  # type: Any
        robot = DummyHardwareRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'calib_scale': [0.5, -1, 1],
                    'calib_offset': [2, 0, -0.5],
                }
            })
        state = RobotState(
            qpos=np.array([1., 1., 1.]), qvel=np.array([1., 1., 1.]))
        robot.calibrate_state(state, 'a')
        np.testing.assert_allclose(state.qpos, [2.5, -1, 0.5])
        np.testing.assert_allclose(state.qvel, [0.5, -1, 1])

    def test_decalibrate_qpos(self):
        """Converts a component state qpos to hardware space."""
        sim_scene = MockSimScene(nq=1)  # type: Any
        robot = DummyHardwareRobotComponent(
            sim_scene,
            groups={
                'a': {
                    'calib_scale': [0.5, -1, 1],
                    'calib_offset': [2, 0, -0.5],
                }
            })
        qpos = robot.decalibrate_qpos(np.array([1., 2., 3.]), 'a')
        np.testing.assert_allclose(qpos, [-2, -2, 3.5])

    def test_timestep(self):
        """Tests advancement of time when doing timesteps."""
        with patch_time(
                'robel.components.robot.hardware_robot.time',
                initial_time=0) as mock_time:
            sim_scene = MockSimScene(nq=1, step_duration=0.5)  # type: Any
            robot = DummyHardwareRobotComponent(sim_scene, groups={})

            self.assertEqual(robot.time, 0)
            robot.do_timestep()
            self.assertAlmostEqual(robot.time, 0.5)
            mock_time.sleep(0.25)
            robot.do_timestep()
            self.assertAlmostEqual(robot.time, 1.0)
            mock_time.sleep(0.6)
            robot.do_timestep()
            self.assertAlmostEqual(robot.time, 1.6)
            robot.reset_time()
            self.assertEqual(robot.time, 0)


if __name__ == '__main__':
    absltest.main()
