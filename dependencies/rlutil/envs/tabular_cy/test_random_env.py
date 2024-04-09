import unittest
import parameterized
import numpy as np

from rlutil.envs.tabular_cy import q_iteration, tabular_env
from rlutil.envs.tabular_cy import q_iteration_py


class QIterationTest(unittest.TestCase):
    def setUp(self):
        self.num_states = 128
        self.env = tabular_env.RandomTabularEnv(num_states=self.num_states, num_actions=3, transitions_per_action=2)
        self.env_selfloop = tabular_env.RandomTabularEnv(num_states=self.num_states, num_actions=3, transitions_per_action=2, self_loop=True)

    def test_num_states(self):
        self.assertEqual(self.env.num_states, self.num_states)

    def test_selfloop(self):
        transitions = self.env_selfloop.transitions(2, 0)
        self.assertEqual(len(transitions), 1)
        self.assertEqual(transitions[2], 1.0)

        transitions = self.env_selfloop.transitions(2, 1)
        self.assertEqual(len(transitions), 2)

    def test_num_transitions(self):
        transitions = self.env.transitions(0, 0)
        self.assertEqual(len(transitions), 2)
        for ns in transitions:
            self.assertAlmostEqual(transitions[ns], 0.5)

    def test_random_rollout(self):
        self.env.reset()
        for _ in range(30):
            #self.env.render()
            self.env.step(np.random.randint(0, self.env.num_actions))

    def test_q_iteration(self):
        params = {
            'num_itrs': 1000,
            'ent_wt': 0.0,
            'discount': 0.95,
        }
        qvals = q_iteration.softq_iteration(self.env, **params)
        self.env.reset()
        rews = 0
        for _ in range(200):
            #self.env_small.render()
            a_qvals = qvals[self.env.get_state()]
            _, rew, _, _ = self.env.step(np.argmax(a_qvals))
            rews += rew
        self.assertGreater(rews, 0.0)

if __name__ == '__main__':
    unittest.main()
