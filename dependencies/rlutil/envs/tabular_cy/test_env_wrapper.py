"""Simple test - make sure there are no silly runtime errors. """
import unittest
import numpy as np
from parameterized import parameterized

from rlutil.envs.tabular_cy import tabular_env, env_wrapper

class TestEpsGreedyWrapper(unittest.TestCase):
    def setUp(self):
        self.env = tabular_env.CliffwalkEnv(10)
        self.env = env_wrapper.StochasticActionWrapper(self.env, eps=0.1)

    def testTransition(self):
        transitions = self.env.transitions(0, 0)
        self.assertAlmostEqual(transitions[0], 0.95)
        self.assertAlmostEqual(transitions[1], 0.05)

        transitions = self.env.transitions(5, 1)
        self.assertAlmostEqual(transitions[6], 0.95)
        self.assertAlmostEqual(transitions[0], 0.05)


class TestAbsorbingStateWrapper(unittest.TestCase):
    def setUp(self):
        self.env = tabular_env.CliffwalkEnv(10)
        self.absorb_env = env_wrapper.AbsorbingStateWrapper(self.env, absorb_reward=123.0)
        self.reward_state = self.env.num_states-1
        self.absorb_state = self.absorb_env.num_states-1

    def testNumStates(self):
        self.assertEqual(self.env.num_states+1, self.absorb_env.num_states)

    def testTransitionAbsorb(self):
        transitions = self.absorb_env.transitions(self.reward_state, 1)
        self.assertEqual(len(transitions), 1)
        self.assertAlmostEqual(transitions[self.absorb_state], 1.0)

    def testTransitionNotAbsorb(self):
        transitions = self.absorb_env.transitions(self.reward_state, 0)
        self.assertEqual(len(transitions), 1)
        self.assertAlmostEqual(transitions[0], 1.0)

    def testReward(self):
        reward = self.absorb_env.reward(self.reward_state, 1, self.absorb_state)
        self.assertAlmostEqual(reward, 123.0)

        reward = self.absorb_env.reward(self.reward_state, 0, 0)
        self.assertAlmostEqual(reward, 0.0)


if __name__ == "__main__":
    unittest.main()
