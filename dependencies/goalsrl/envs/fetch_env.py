"""
A GoalEnv which wraps the gym Fetch environments

Observation Space: Varies 
Goal Space: Varies
Action Space (3 dim): End-Effector Position Control
"""

from goalsrl.envs.gymenv_wrapper import GymGoalEnvWrapper
from gym.envs.robotics import fetch

configs = {
    'reach': fetch.reach.FetchReachEnv,
    'push': fetch.push.FetchPushEnv,
    'slide': fetch.slide.FetchSlideEnv,
    'pick_and_place': fetch.pick_and_place.FetchPickAndPlaceEnv,
}

class FetchEnv(GymGoalEnvWrapper):
    def __init__(self, env_type='reach'):
        assert env_type in configs
        env = configs[env_type](reward_type='dense')
        self.distance_threshold = env.distance_threshold
        super(FetchEnv, self).__init__(env, observation_key='observation',
        goal_key='achieved_goal', state_goal_key='achieved_goal', use_internal_rewards=True)
    
    def goal_distance(self, state, goal_state):
        # Uses distance in state_goal_key to determine distance (useful for images)
        state_sgoal = self._extract_sgoal(state)
        goal_sgoal = self._extract_sgoal(goal_state)

        
        distances = -1 * self.base_env.compute_reward(state_sgoal, goal_sgoal, dict())
        return distances