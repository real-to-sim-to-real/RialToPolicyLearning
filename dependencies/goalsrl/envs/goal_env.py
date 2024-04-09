import gym
import numpy as np
from collections import OrderedDict

class GoalEnv(gym.Env):
    """
    A GoalEnv is a modified Gym environment designed for goal-reaching tasks.

    One of the main deviations from the standard Gym abstraction is the separation
    of state from observation. The step() method always returns states, and
    observations can be obtained using a separate observation() method. 

    This change makes it easy to check for goal status, because whether a state
    reaches a goal is not always computable from the observation.

    The API consists of the following:
        GoalEnv.state_space
        GoalEnv.goal_space
        GoalEnv.reset()
            Resets the environment and returns the *state*
        GoalEnv.step(action)
            Runs 1 step of simulation and returns (state, 0, done, infos)
        GoalEnv.observation(state)
            Returns the observation for a given state
        GoalEnv.extract_goal(state)
            Returns the goal representation for a given state
    """
    def __init__(self, goal_metric='euclidean', goal_threshold=0.01):
        super(GoalEnv, self).__init__()
        self.goal_metric = goal_metric
        self.goal_space = self.observation_space
        self.state_space = self.observation_space

    def reset(self):
        """
        Resets the environment and returns a state vector

        Returns:
            The initial state
        """
        raise NotImplementedError()

    def step(self, a):
        """
        Runs 1 step of simulation

        Returns:
            A tuple containing:
                next_state
                reward (always 0)
                done
                infos
        """
        raise NotImplementedError()

    def observation(self, state):
        """
        Returns the observation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise NotImplementedError()
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        raise NotImplementedError()

    def goal_distance(self, state, goal_state):
        if self.goal_metric == 'euclidean':
            diff = self.extract_goal(state)-self.extract_goal(goal_state)
            return np.linalg.norm(diff, axis=-1) 
        else:
            raise ValueError('Unknown goal metric %s' % self.goal_metric)

    def sample_goal(self):
        return self.goal_space.sample()

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Gets things to log

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        Returns:
            An Ordered Dictionary containing k,v pairs to be logged
        """

        distances = np.array([self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        return OrderedDict([
            ('mean final l2 dist', np.mean(distances[:,-1])),
            ('median final l2 dist', np.median(distances[:,-1])),
        ])