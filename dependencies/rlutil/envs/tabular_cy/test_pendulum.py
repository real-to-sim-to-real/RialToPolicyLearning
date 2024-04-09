import unittest
import parameterized
import numpy as np

from rlutil.envs.tabular_cy import q_iteration, tabular_env
from rlutil.envs.tabular_cy import q_iteration_py


class QIterationTest(unittest.TestCase):
    def setUp(self):
        self.env = tabular_env.InvertedPendulum(state_discretization=512)
        self.env_small = tabular_env.InvertedPendulum(state_discretization=32, action_discretization=5)

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
        qvals = q_iteration.softq_iteration(self.env_small, **params)
        self.env_small.reset()
        for _ in range(200):
            #self.env_small.render()
            a_qvals = qvals[self.env_small.get_state()]
            _, rew, _, _ = self.env_small.step(np.argmax(a_qvals))
            #print(rew)
        self.assertEqual(rew, 1.0)

if __name__ == '__main__':
    unittest.main()
