import numpy as np

class Policy(object):
    def __init__(self):
        pass

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return self.act_vectorized(state)[0]

    def act_vectorized(self, state):
        raise NotImplementedError()
    
    def reset(self):
        pass

class GoalConditionedPolicy(object):
    def __init__(self):
        pass

    def act(self, state, goal_state):
        state = np.expand_dims(state, axis=0)
        goal_state = np.expand_dims(goal_state, axis=0)
        return self.act_vectorized(state, goal_state)[0]

    def act_vectorized(self, state, goal_state):
        raise NotImplementedError()
    
    def reset(self):
        pass

