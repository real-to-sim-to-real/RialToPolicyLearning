import numpy as np
from goalsrl.envs import goal_env
import gym

class GymGoalEnvWrapper(goal_env.GoalEnv):
    
    """
    
    A wrapper around multiworld (github.com/vitchyr/multiworld) environments. and gym GoalEnvs

    """
    
    def __init__(self, base_env, observation_key='observation', goal_key='achieved_goal', state_goal_key='achieved_goal', use_internal_rewards=False):
        super(GymGoalEnvWrapper, self).__init__()
        self.base_env = base_env

        self.action_space = self.base_env.action_space
        
        all_space = self.base_env.observation_space
        
        self.obs_key = observation_key
        self.observation_space = all_space.spaces[observation_key]
        self.goal_key = goal_key
        self.goal_space = all_space.spaces[goal_key]
        self.sgoal_key = state_goal_key
        self.sgoal_space = all_space.spaces[state_goal_key]

        # concat observation and goal to get the state
        obs_low = self.observation_space.low.flatten()
        goal_low = self.goal_space.low.flatten()
        sgoal_low = self.sgoal_space.low.flatten()
        state_low = np.r_[obs_low, goal_low, sgoal_low]

        obs_hi = self.observation_space.high.flatten()
        goal_hi = self.goal_space.high.flatten()
        sgoal_hi = self.sgoal_space.high.flatten()
        state_high = np.r_[obs_hi, goal_hi, sgoal_hi]

        self.state_space = gym.spaces.Box(low=state_low, high=state_high)

        self.obs_dims = obs_low.shape[0]
        self.goal_dims = goal_low.shape[0]
        self.sgoal_dims = sgoal_low.shape[0]

        self.use_internal_rewards = use_internal_rewards

    def _base_obs_to_state(self, base_obs):
        obs = base_obs[self.obs_key].flatten()
        goal = base_obs[self.goal_key].flatten()
        sgoal = base_obs[self.sgoal_key].flatten()
        return np.r_[obs, goal, sgoal]

    def reset(self):
        """
        Resets the environment and returns a state vector

        Returns:
            The initial state
        """
        base_obs = self.base_env.reset()
        return self._base_obs_to_state(base_obs)

    def render(self):
        return self.base_env.render()
        
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
        ns, reward, done, infos = self.base_env.step(a)
        ns = self._base_obs_to_state(ns)
        return ns, reward, done, infos

    def observation(self, state):
        """
        Returns the observation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        obs = state[...,:self.obs_dims]
        return obs.reshape(obs.shape[:len(obs.shape)-1]+self.observation_space.shape)
    
    def extract_goal(self, state):
        """
        Returns the goal representation for a given state

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        goal = state[...,self.obs_dims:self.obs_dims+self.goal_dims]
        return goal.reshape(goal.shape[:len(goal.shape)-1]+self.goal_space.shape)

    def _extract_sgoal(self, state):
        """
        Returns the state goal representation for a given state (internal)

        Args:
            state: A numpy array representing state
        Returns:
            obs: A numpy array representing observations
        """
        sgoal = state[..., self.obs_dims + self.goal_dims:]
        return sgoal.reshape(sgoal.shape[:len(sgoal.shape)-1]+self.sgoal_space.shape)

    def sample_goal(self):
        """
        Samples a goal state (of type self.state_space.sample()) using 'desired_goal'
        
        """

        desired_key = self.goal_key.replace('achieved', 'desired')
        desired_state_key = self.sgoal_key.replace('achieved', 'desired')

        base_obs = self.base_env.reset()
        obs = (10 + self.observation_space.sample()).flatten() # Placeholder - shouldn't actually be used
        goal = base_obs[desired_key].flatten()
        sgoal = base_obs[desired_state_key].flatten()

        return np.r_[obs, goal, sgoal]
    
    def goal_distance(self, state, goal_state):
        # Uses distance in state_goal_key to determine distance (useful for images)
        state_sgoal = self._extract_sgoal(state)
        goal_sgoal = self._extract_sgoal(goal_state)

        if self.use_internal_rewards and hasattr(self.base_env, 'compute_reward'):
            distances = np.abs(np.array([
                self.base_env.compute_reward(achieved, desired, dict()) for achieved, desired in zip(state_sgoal, goal_sgoal)
            ]))
        else:
            distances = np.linalg.norm(state_sgoal - goal_sgoal, axis=-1)
        
        return distances
        