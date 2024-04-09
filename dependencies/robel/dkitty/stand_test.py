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

"""Unit tests for DKitty walk tasks."""

from absl.testing import absltest
from absl.testing import parameterized
import gym
import numpy as np

from robel.dkitty.stand import (DKittyStandFixed, DKittyStandRandom,
                                     DKittyStandRandomDynamics)
# pylint: disable=no-member


@parameterized.parameters(
    ('DKittyStandFixed-v0', DKittyStandFixed),
    ('DKittyStandRandom-v0', DKittyStandRandom),
    ('DKittyStandRandomDynamics-v0', DKittyStandRandomDynamics),
)
class DKittyStandTest(absltest.TestCase):
    """Unit test class for RobotEnv."""

    def test_gym_make(self, env_id, env_cls):
        """Accesses the sim, model, and data properties."""
        env = gym.make(env_id)
        self.assertIsInstance(env.unwrapped, env_cls)

    def test_spaces(self, _, env_cls):
        """Checks the observation, action, and state spaces."""
        env = env_cls()
        observation_size = np.sum([
            3,  # root_qpos
            3,  # root_euler
            12,  # kitty_qpos
            3,  # root_vel
            3,  # root_angular_vel
            12,  # kitty_qvel
            12,  # last_action
            1,  # upright
            12,  # pose error
        ])
        self.assertEqual(env.observation_space.shape, (observation_size,))
        self.assertEqual(env.action_space.shape, (12,))
        self.assertEqual(env.state_space['root_pos'].shape, (3,))
        self.assertEqual(env.state_space['root_euler'].shape, (3,))
        self.assertEqual(env.state_space['root_vel'].shape, (3,))
        self.assertEqual(env.state_space['root_angular_vel'].shape, (3,))
        self.assertEqual(env.state_space['kitty_qpos'].shape, (12,))
        self.assertEqual(env.state_space['kitty_qvel'].shape, (12,))

    def test_reset_step(self, _, env_cls):
        """Checks that resetting and stepping works."""
        env = env_cls()
        env.reset()
        env.step(env.action_space.sample())


if __name__ == '__main__':
    absltest.main()
