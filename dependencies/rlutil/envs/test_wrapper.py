"""Simple test - make sure there are no silly runtime errors. """
import unittest
import numpy as np
from parameterized import parameterized

from rlutil.envs.tabular_cy import tabular_env
from rlutil.envs import wrappers

class TestTimeLimitWrapper(unittest.TestCase):
    def setUp(self):
        self.env = tabular_env.CliffwalkEnv(10)
        self.T = 12
        self.env = wrappers.TimeLimitWrapper(self.env, time_limit=self.T)

    def test_time_limit(self):
        self.env.reset()
        for _ in range(self.T-1):
            obs, rew, done, infos = self.env.step(0)
            self.assertFalse(done)
        obs, rew, done, infos = self.env.step(0)
        self.assertTrue(done)


if __name__ == "__main__":
    unittest.main()
