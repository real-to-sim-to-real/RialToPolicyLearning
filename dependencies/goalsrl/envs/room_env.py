"""
A GoalEnv which wraps my room_world environments

Observation Space (2 dim): Position 
Goal Space (2 dim): Position
Action Space (2 dim): Position Control
"""

import numpy as np
from room_world.pointmass import PMEnv, pointmass_camera_config
from goalsrl.envs.gymenv_wrapper import GymGoalEnvWrapper
from goalsrl.envs.env_utils import DiscretizedActionEnv, ImageEnv


from collections import OrderedDict
from multiworld.envs.env_util import create_stats_ordered_dict

class PointmassGoalEnv(GymGoalEnvWrapper):
    def __init__(self, room_type='empty', fixed_start=False, fixed_goal=False, images=False, image_kwargs=None, env_kwargs=None):
        
        assert room_type in ['empty', 'wall', 'rooms']
        config = dict(
            room_type=room_type,
            potential_type="none",
            shaped=True,
            max_path_length=50,
            use_state_images=False,
            use_goal_images=False,
        )

        if fixed_start:
            config['start_config'] = (np.array([-0.33,-0.33]), np.array([-0.27,-0.27])) # Start at / around (-0.3, -0.3)

        if fixed_goal:
            config['goal_config'] = (np.array([0.27,0.27]), np.array([0.33,0.33])) # End at / around (0.3, 0.3)
        
        if env_kwargs is not None:
            config.update(env_kwargs)

        env = PMEnv(**config)
        self.base_env = env
        
        if images:
            config = dict(init_camera=pointmass_camera_config, imsize=84, normalize=True, channels_first=True, )
            if image_kwargs is not None:
                config.update(image_kwargs)
            env = ImageEnv(env, **config)

        super(PointmassGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='state_achieved_goal'
        )
    
    def shaped_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        return np.array([
            self.base_env.room.get_shaped_distance(achieved_goals[i], desired_goals[i])
            for i in range(achieved_goals.shape[0])
        ])

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        euclidean_distances = np.array([self.goal_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        shaped_distances = np.array([self.shaped_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final l2 distance', euclidean_distances[:,-1]),
            ('final shaped distance', shaped_distances[:,-1]),
            ('l2 distance', euclidean_distances),
            ('shaped_distances', shaped_distances),
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                ))
            
        return statistics