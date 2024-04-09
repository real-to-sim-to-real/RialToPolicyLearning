from rlutil.envs.gridcraft.grid_env import GridEnv, ACT_RIGHT
from rlutil.envs.gridcraft.mazes import MAZE_LAVA
from rlutil.envs.gridcraft.wrappers import RandomObsWrapper

import unittest


class GridEnvCyTest(unittest.TestCase):
    def testRun(self):
        env = GridEnv(MAZE_LAVA, teps=0.2)
        env = RandomObsWrapper(env, 5)
        env.reset()
        for _ in range(20):
            env.step(ACT_RIGHT)


if __name__ == "__main__":
    unittest.main()