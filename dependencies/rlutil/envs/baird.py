import numpy as np
from gym.envs.toy_text import discrete

SOLID = 1
DASHED = 0


class Baird(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        nS = 7
        nA = 2

        P = {}

        for s in range(nS):
            P[s] = {a: [] for a in range(nA)}
            # solid action always go to star center
            P[s][SOLID] = [(1.0, 6, 0.0, False)]
            # dashed action goes to star corners uniformly
            P[s][DASHED] = [(1.0/6, ns, 0.0, False) for ns in range(nS-1)]

        # Initial state distribution is uniform
        isd = np.ones(nS, dtype='double')/nS

        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(Baird, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return
        # print("render")
        print("current state: ", self.state)
