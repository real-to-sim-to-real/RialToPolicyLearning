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

"""Unit tests for D'Claw pose tasks."""

from absl.testing import absltest
from absl.testing import parameterized
import gym
import numpy as np

from robel.dclaw.pose import (DClawPoseFixed, DClawPoseRandom,
                                   DClawPoseRandomDynamics)
# pylint: disable=no-member


@parameterized.parameters(
    ('DClawPoseFixed-v0', DClawPoseFixed),
    ('DClawPoseRandom-v0', DClawPoseRandom),
    ('DClawPoseRandomDynamics-v0', DClawPoseRandomDynamics),
)
class DClawPoseTest(absltest.TestCase):
    """Unit test class for RobotEnv."""

    def test_gym_make(self, env_id, env_cls):
        """Accesses the sim, model, and data properties."""
        env = gym.make(env_id)
        self.assertIsInstance(env.unwrapped, env_cls)

    def test_spaces(self, _, env_cls):
        """Checks the observation, action, and state spaces."""
        env = env_cls()
        observation_size = np.sum([
            9,  # qpos
            9,  # last_action
            9,  # qpos_error
        ])
        self.assertEqual(env.observation_space.shape, (observation_size,))
        self.assertEqual(env.action_space.shape, (9,))
        self.assertEqual(env.state_space['qpos'].shape, (9,))
        self.assertEqual(env.state_space['qvel'].shape, (9,))

    def test_reset_step(self, _, env_cls):
        """Checks that resetting and stepping works."""
        env = env_cls()
        env.reset()
        env.step(env.action_space.sample())


if __name__ == '__main__':
    absltest.main()
