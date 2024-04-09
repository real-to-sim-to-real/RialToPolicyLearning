import unittest
import parameterized
import numpy as np

from rlutil.envs.tabular_cy import q_iteration, tabular_env
from rlutil.envs.tabular import q_iteration as q_iteration_py


class QIterationTest(unittest.TestCase):
  def setUp(self):
    self.env = tabular_env.CliffwalkEnv(num_states=3, transition_noise=0.01)

  def test_qiteration(self):
    params = {
        'num_itrs': 50,
        'ent_wt': 1.0,
        'discount': 0.99,
    }
    qvals_py = q_iteration_py.softq_iteration(self.env, **params)
    qvals_cy = q_iteration.softq_iteration(self.env, **params)
    self.assertTrue(np.allclose(qvals_cy, qvals_py))

  def test_qevaluation_noent(self):
    env = tabular_env.CliffwalkEnv(num_states=2, transition_noise=0.00)
    params = {
        'num_itrs': 100,
        'ent_wt': 0.0,
        'discount': 0.5,
    }
    q_values = np.zeros((env.num_states, env.num_actions))
    q_values[:, 1] = 1e10
    returns, _ = q_iteration.softq_evaluation(env, q_values, **params)
    self.assertAlmostEqual(returns, 0.66666666)

  def test_qevaluation_ent(self):
    env = tabular_env.CliffwalkEnv(num_states=2, transition_noise=0.00)
    params = {
        'num_itrs': 100,
        'ent_wt': 0.001,
        'discount': 0.5,
    }
    q_values = np.zeros((env.num_states, env.num_actions))
    q_values[:, 1] = 1e10
    returns, _ = q_iteration.softq_evaluation(env, q_values, **params)
    self.assertAlmostEqual(returns, 0.66666666)

  def test_visitations(self):
    env = tabular_env.CliffwalkEnv(num_states=3, transition_noise=0.00)
    params = {
        'num_itrs': 50,
        'ent_wt': 0.0,
        'discount': 0.99,
    }
    qvals_py = q_iteration_py.softq_iteration(env, **params)

    visitations = q_iteration_py.compute_visitation(env, qvals_py, ent_wt=0.0, env_time_limit=1)
    s_visitations = np.sum(visitations, axis=1)
    tru_visits = np.array([1, 0, 0]) 
    self.assertTrue(np.allclose(tru_visits, s_visitations))

    visitations = q_iteration_py.compute_visitation(env, qvals_py, ent_wt=0.0, env_time_limit=3)
    s_visitations = np.sum(visitations, axis=1)
    tru_visits = np.array([1, 1, 1]) / 3.0
    self.assertTrue(np.allclose(tru_visits, s_visitations))

    visitations = q_iteration_py.compute_visitation(env, qvals_py, ent_wt=0.0, env_time_limit=5)
    s_visitations = np.sum(visitations, axis=1)
    tru_visits = np.array([2, 2, 1]) / 5.0
    self.assertTrue(np.allclose(tru_visits, s_visitations))

if __name__ == '__main__':
  unittest.main()
