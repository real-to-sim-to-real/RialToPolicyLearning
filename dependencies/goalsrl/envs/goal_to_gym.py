import gym
from multiworld.core.serializable import Serializable
import numpy as np
from rllab.misc import logger

class GoalEnvWrapper(gym.Env):
    def __init__(self, base_env):
        # Serializable.__init__(self, base_env)
        super().__init__()

        self.env = base_env
        self.desired_goal = base_env.sample_goal()

        obs_low = self.env.observation_space.low #.flatten()
        goal_low = self.env.goal_space.low #.flatten()
        state_low = np.r_[obs_low, goal_low]

        obs_hi = self.env.observation_space.high #.flatten()
        goal_hi = self.env.goal_space.high #.flatten()
        state_high = np.r_[obs_hi, goal_hi]

        self.observation_space = gym.spaces.Box(low=state_low, high=state_high)
        self.action_space = self.env.action_space

    def state_to_obs(self, state, desired_state):
        obs = self.env.observation(state) #.flatten()
        goal = self.env.extract_goal(desired_state) #.flatten()
        return np.r_[obs, goal]
    
    def reset(self):
        self.desired_state = self.env.sample_goal()
        state = self.env.reset()
        self.last_distance = self.env.goal_distance(state, self.desired_state)
        return self.state_to_obs(state, self.desired_state)
    
    def step(self, action):
        ns, _, done, infos = self.env.step(action)
        
        new_distance = self.env.goal_distance(ns, self.desired_state)
        reward = self.last_distance - new_distance
        self.last_distance = new_distance

        observation = self.state_to_obs(ns, self.desired_state)
        infos['original_state'] = ns
        infos['original_goal'] = self.desired_state
        return observation, reward, done, infos
    
    def render(self):
        return self.env.render()
    
    def get_param_values(self):
        return dict()
    
    def set_param_values(self, d):
        return
    
    def log_diagnostics(self, paths):
        trajectories = np.array([path['env_infos']['original_state'] for path in paths])
        goals = np.array([path['env_infos']['original_goal'][0] for path in paths])
        for k,v in self.env.get_diagnostics(trajectories, goals).items():
            logger.record_tabular(k, v)


class EpsilonGoalEnvWrapper(gym.Env):
    def __init__(self, base_env, threshold):
        # Serializable.__init__(self, base_env)
        super().__init__()

        self.env = base_env
        self.desired_goal = base_env.sample_goal()
        self.threshold = threshold

        obs_low = self.env.observation_space.low #.flatten()
        goal_low = self.env.goal_space.low #.flatten()
        state_low = np.r_[obs_low, goal_low]
        if len(state_low.shape) == 3:
            state_low = np.swapaxes(state_low, 0, 2)

        obs_hi = self.env.observation_space.high #.flatten()
        goal_hi = self.env.goal_space.high #.flatten()
        state_high = np.r_[obs_hi, goal_hi]
        if len(state_low.shape) == 3:
            state_high = np.swapaxes(state_high, 0, 2)

        self.observation_space = gym.spaces.Box(low=state_low, high=state_high)
        self.action_space = self.env.action_space

    def state_to_obs(self, state, desired_state):
        obs = self.env.observation(state) #.flatten()
        goal = self.env.extract_goal(desired_state) #.flatten()
        return np.swapaxes(np.r_[obs, goal], 0, 2)
    
    def reset(self):
        self.desired_state = self.env.sample_goal()
        state = self.env.reset()
        self.last_distance = self.env.goal_distance(state, self.desired_state)
        return self.state_to_obs(state, self.desired_state)
    
    def step(self, action):
        ns, _, done, infos = self.env.step(action)
        
        new_distance = self.env.goal_distance(ns, self.desired_state)
        reward = new_distance < self.threshold
        self.last_distance = new_distance

        observation = self.state_to_obs(ns, self.desired_state)
        infos['original_state'] = ns
        infos['original_goal'] = self.desired_state
        infos['goal_distance'] = new_distance
        infos['success'] = new_distance < self.threshold

        return observation, reward, done, infos
    
    def render(self):
        return self.env.render()
    
    def get_param_values(self):
        return dict()
    
    def set_param_values(self, d):
        return
    
    def log_diagnostics(self, paths):
        trajectories = np.array([path['env_infos']['original_state'] for path in paths])
        goals = np.array([path['env_infos']['original_goal'][0] for path in paths])
        for k,v in self.env.get_diagnostics(trajectories, goals).items():
            logger.record_tabular(k, v)
        logger.record_tabular('EvalGreedy avg final dist', np.mean([path['env_infos']['goal_distance'][-1] for path in paths]))
        logger.record_tabular('EvalGreedy success ratio', np.mean([path['env_infos']['success'][-1] for path in paths]))