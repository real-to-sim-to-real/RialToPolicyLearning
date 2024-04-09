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

"""Unit tests for RobotEnv."""

import collections

from absl.testing import absltest
from absl.testing.absltest import mock
import numpy as np

from robel.robot_env import RobotEnv
from robel.utils.testing.mock_sim_scene import patch_sim_scene


class TestEnv(RobotEnv):
    """Dummy environment for testing RobotEnv."""

    def __init__(self, nq=1, **kwargs):
        # Replace SimScene with a mocked object.
        with patch_sim_scene('robel.robot_env.SimScene', nq=nq):
            super().__init__('', **kwargs)

    def _reset(self):
        pass

    def _step(self, action):
        pass

    def get_obs_dict(self):
        return {}

    def get_reward_dict(self, action, obs_dict):
        return {}

    def get_score_dict(self, obs_dict, reward_dict):
        return {}


class RobotEnvTest(absltest.TestCase):
    """Unit test class for RobotEnv."""

    # pylint: disable=protected-access

    def test_sim_properties(self):
        """Accesses the sim, model, and data properties."""
        test = TestEnv()
        self.assertIsNotNone(test.sim)
        self.assertIsNotNone(test.model)
        self.assertIsNotNone(test.data)
        # Check that the random state is initialized.
        self.assertIsNotNone(test.np_random)

    def test_init_observation_space(self):
        """Initializes the observation space."""
        test = TestEnv()
        test._initialize_observation_space = mock.Mock(return_value=1)
        self.assertEqual(test._initialize_observation_space.call_count, 0)

        # Check that the observation space gets correctly initialized.
        self.assertEqual(test.observation_space, 1)
        self.assertEqual(test._initialize_observation_space.call_count, 1)

        # Check that the observation space does not get re-initialized.
        self.assertEqual(test.observation_space, 1)
        self.assertEqual(test._initialize_observation_space.call_count, 1)

    def test_default_observation_space(self):
        """Tests the default observation space."""
        test = TestEnv()
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('a', [1, 2])]))
        self.assertEqual(test.observation_space.shape, (2,))

    def test_dict_observation_space(self):
        """Tests the default observation space."""
        test = TestEnv(use_dict_obs=True)
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('a', [1, 2])]))
        self.assertEqual(test.observation_space.spaces['a'].shape, (2,))

    def test_init_action_space(self):
        """Initializes the action space."""
        test = TestEnv()
        test._initialize_action_space = mock.Mock(return_value=1)
        self.assertEqual(test._initialize_action_space.call_count, 0)

        # Check that the action space gets correctly initialized.
        self.assertEqual(test.action_space, 1)
        self.assertEqual(test._initialize_action_space.call_count, 1)

        # Check that the action space does not get re-initialized.
        self.assertEqual(test.action_space, 1)
        self.assertEqual(test._initialize_action_space.call_count, 1)

    def test_default_action_space(self):
        """Tests the default action space."""
        test = TestEnv(nq=5)
        self.assertEqual(test.action_space.shape, (5,))
        np.testing.assert_array_equal(test.action_space.low, [-1] * 5)
        np.testing.assert_array_equal(test.action_space.high, [1] * 5)

    def test_init_state_space(self):
        """Initializes the state space."""
        test = TestEnv()
        test._initialize_state_space = mock.Mock(return_value=1)
        self.assertEqual(test._initialize_state_space.call_count, 0)

        # Check that the state space gets correctly initialized.
        self.assertEqual(test.state_space, 1)
        self.assertEqual(test._initialize_state_space.call_count, 1)

        # Check that the state space does not get re-initialized.
        self.assertEqual(test.state_space, 1)
        self.assertEqual(test._initialize_state_space.call_count, 1)

    def test_default_state_space(self):
        """Tests the default state space."""
        test = TestEnv(nq=4)
        self.assertEqual(test.state_space[0].shape, (4,))
        self.assertEqual(test.state_space[1].shape, (4,))

    def test_reset(self):
        """Performs a reset."""
        test = TestEnv()
        test._reset = mock.Mock()
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('a', [1, 1])]))
        obs = test.reset()

        self.assertIsNone(test.last_action)
        self.assertDictEqual(test.last_obs_dict, {'a': [1, 1]})
        self.assertIsNone(test.last_reward_dict)
        self.assertIsNone(test.last_score_dict)
        self.assertFalse(test.is_done)
        self.assertEqual(test.step_count, 0)

        self.assertEqual(test._reset.call_count, 1)
        np.testing.assert_array_equal(obs, [1, 1])

    def test_reset_dict(self):
        """Performs a reset using dictionary observations."""
        test = TestEnv(use_dict_obs=True)
        test._reset = mock.Mock()
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('a', [1, 1])]))
        obs = test.reset()

        self.assertEqual(test._reset.call_count, 1)
        self.assertEqual(test.get_obs_dict.call_count, 1)
        np.testing.assert_array_equal(obs['a'], [1, 1])

    def test_step(self):
        """Checks that step calls its subcomponents."""
        test = TestEnv()
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([
                ('o1', [0]),
                ('o2', [1, 2]),
            ]))
        test.get_reward_dict = mock.Mock(return_value={
            'r1': np.array(1),
            'r2': np.array(2),
        })
        test.get_score_dict = mock.Mock(return_value={
            's1': np.array(1),
        })
        test.get_done = mock.Mock(return_value=np.array(False))

        obs, reward, done, info = test.step([0])

        np.testing.assert_array_equal(test.last_action, [0])
        self.assertDictEqual(test.last_obs_dict, {'o1': [0], 'o2': [1, 2]})
        self.assertDictEqual(test.last_reward_dict, {'r1': 1, 'r2': 2})
        self.assertDictEqual(test.last_score_dict, {'s1': 1})
        self.assertFalse(test.is_done)
        self.assertEqual(test.step_count, 1)

        # Check that step calls its sub-methods exactly once.
        self.assertEqual(test.get_obs_dict.call_count, 1)
        self.assertEqual(test.get_reward_dict.call_count, 1)
        self.assertEqual(test.get_score_dict.call_count, 1)
        self.assertEqual(test.get_done.call_count, 1)

        np.testing.assert_array_equal(obs, [0, 1, 2])
        self.assertEqual(reward, 3)
        self.assertFalse(done)
        self.assertListEqual(
            sorted(info.keys()), [
                'obs/o1', 'obs/o2', 'reward/r1', 'reward/r2', 'reward/total',
                'score/s1'
            ])
        np.testing.assert_array_equal(info['obs/o1'], [0])
        np.testing.assert_array_equal(info['obs/o2'], [1, 2])
        np.testing.assert_array_equal(info['reward/r1'], 1)
        np.testing.assert_array_equal(info['reward/r2'], 2)
        np.testing.assert_array_equal(info['reward/total'], 3)
        np.testing.assert_array_equal(info['score/s1'], 1)

    def test_step_dict(self):
        """Checks the output of step using dictionary observations."""
        test = TestEnv(use_dict_obs=True)
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([
                ('o1', [0]),
                ('o2', [1, 2]),
            ]))
        test.get_reward_dict = mock.Mock(return_value={
            'r1': np.array(1),
            'r2': np.array(2),
        })

        obs, _, _, _ = test.step([0])
        self.assertListEqual(sorted(obs.keys()), ['o1', 'o2'])
        np.testing.assert_array_equal(obs['o1'], [0])
        np.testing.assert_array_equal(obs['o2'], [1, 2])

    def test_get_obs_subset(self):
        """Tests `_get_obs` flattening a subset of keys."""
        test = TestEnv(observation_keys=['b', 'd'])
        test.get_obs_dict = mock.Mock(return_value={
            'a': [0],
            'b': [1, 2],
            'c': [3, 4],
            'd': [5],
        })
        np.testing.assert_array_equal(test._get_obs(), [1, 2, 5])

    def test_get_obs_subset_dict(self):
        """Tests `_get_obs` flattening a subset of keys."""
        test = TestEnv(observation_keys=['b', 'd'], use_dict_obs=True)
        test.get_obs_dict = mock.Mock(return_value={
            'a': [0],
            'b': [1, 2],
            'c': [3, 4],
            'd': [5],
        })
        obs = test._get_obs()
        self.assertListEqual(sorted(obs.keys()), ['b', 'd'])
        np.testing.assert_array_equal(obs['b'], [1, 2])
        np.testing.assert_array_equal(obs['d'], [5])

    def test_get_total_reward(self):
        """Tests `_get_total_reward` summing a subset of keys."""
        test = TestEnv(reward_keys=['b', 'c'])
        test_rewards = {
            'a': 1,
            'b': 3,
            'c': 5,
            'd': 7,
        }
        self.assertEqual(test._get_total_reward(test_rewards), 8)

    def test_get_done(self):
        """Tests default behavior of `get_done`."""
        test = TestEnv(reward_keys=['b', 'c'])
        test_rewards = {
            'a': np.array([1, 2]),
        }
        np.testing.assert_array_equal(
            test.get_done({}, test_rewards), [False, False])

    def test_last_action(self):
        """Tests reading the last action."""
        test = TestEnv(nq=3)
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('o', [0])]))
        test.get_reward_dict = mock.Mock(return_value={'r': np.array(0)})

        self.assertIsNone(test.last_action)
        np.testing.assert_array_equal(test._get_last_action(), [0, 0, 0])

        test.step([1, 1, 1])
        np.testing.assert_array_equal(test._get_last_action(), [1, 1, 1])
        self.assertIs(test.last_action, test._get_last_action())

    def test_sticky_actions(self):
        """Tests sticky actions."""
        test = TestEnv(nq=3, sticky_action_probability=1)
        test.get_obs_dict = mock.Mock(
            return_value=collections.OrderedDict([('o', [0])]))
        test.get_reward_dict = mock.Mock(return_value={'r': np.array(0)})
        test._step = mock.Mock()

        test.last_action = np.ones(3)
        test.step([0, 0, 0])

        args, _ = test._step.call_args_list[0]
        np.testing.assert_array_equal(args[0], np.ones(3))


if __name__ == '__main__':
    absltest.main()
