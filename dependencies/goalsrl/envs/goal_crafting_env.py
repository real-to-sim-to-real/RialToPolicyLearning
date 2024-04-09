# From Soroush: https://github.com/vitchyr/multiworld/blob/ec5f53914341ce188b7327f38a3c2b8a5b7dfad0/multiworld/envs/mujoco/sawyer_xyz/sawyer_push_nips.py
"""
A GoalEnv which wraps the multiworld SawyerPush environments

Observation Space (4 dim): EE + Puck Position 
Goal Space (4 dim): EE + Puck Position
Action Space (2 dim): EE Position Control
"""

from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)
import copy

from multiworld.core.multitask_env import MultitaskEnv
import matplotlib.pyplot as plt
import os.path as osp
from goalsrl.envs.gymenv_wrapper import GymGoalEnvWrapper
from goalsrl.envs.env_utils import DiscretizedActionEnv, ImageEnv

from crafting_env.multiworld_crafting_env import MultiworldCraftingEnv

class CraftingGoalEnv(GymGoalEnvWrapper):
    def __init__(self, fixed_start=False, images=False, image_kwargs=None):
        env = MultiworldCraftingEnv(size=[10,10], res=3, state_obs=True, few_obj=True, 
                                    append_init_state_to_goal=False, fixed_init_state=True)

        super(CraftingGoalEnv, self).__init__(
            env, observation_key='observation', goal_key='achieved_goal', state_goal_key='achieved_goal'
        )
    
    
    def goal_distance(self, states, goal_states):
        return self.distance(states, goal_states)
    
    def agent_distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.sum(np.abs(diff[:,0:2]), axis=-1)
    
    def distance(self, states, goal_states):
        achieved_goals = self._extract_sgoal(states)
        desired_goals = self._extract_sgoal(goal_states)
        diff = achieved_goals - desired_goals
        return np.sum(np.abs(diff), axis=-1)

    def get_diagnostics(self, trajectories, desired_goal_states):
        """
        Logs things

        Args:
            trajectories: Numpy Array [# Trajectories x Max Path Length x State Dim]
            desired_goal_states: Numpy Array [# Trajectories x State Dim]

        """
        agent_distances = np.array([self.agent_distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])
        distances = np.array([self.distance(trajectories[i], np.tile(desired_goal_states[i], (trajectories.shape[1],1))) for i in range(trajectories.shape[0])])

        agent_movement = self.agent_distance(trajectories[:,0], trajectories[:, -1])
        
        achieved_task_successes = np.zeros((trajectories.shape[0],9))
        goal_task_successes = np.zeros((trajectories.shape[0],9))
        for i in range(trajectories.shape[0]):
            init_state = self.observation(trajectories[i, 0])
            final_state = self.observation(trajectories[i, -1])
            goal_sgoal = self._extract_sgoal(desired_goal_states[i])
            achieved_tasks = self.base_env.eval_tasks(init_state, final_state)
            goal_tasks =  self.base_env.eval_tasks(init_state,goal_sgoal )
            achieved_task_successes[i] = achieved_tasks
            goal_task_successes[i] = goal_tasks
        with np.errstate(divide='ignore', invalid='ignore'):
            goal_task_successes = np.array([np.mean(np.nan_to_num(achieved_task_successes[goal_task_successes[:,t]>0,t]/goal_task_successes[goal_task_successes[:,t]>0,t]), axis=0) for t in range(9)])        
        achieved_task_successes = np.mean(achieved_task_successes, axis=0)
        
        statistics = OrderedDict()
        for stat_name, stat in [
            ('final agent distance', agent_distances[:,-1]),
            ('final distance', distances[:,-1]),
            ('agent movement', agent_movement),
            ('agent success', agent_distances[:, -1]< 1)
        ]:
            statistics.update(create_stats_ordered_dict(
                    stat_name,
                    stat,
                    always_show_all_stats=True,
                exclude_max_min=True
                ))
        for task_ind , task in enumerate(self.base_env.tasks):
            statistics['commanded_'+task] =goal_task_successes[task_ind]
            statistics['achieved_'+task] =achieved_task_successes[task_ind]
        return statistics
        

def main():
    from rlutil.logging import logger
    e = CraftingGoalEnv()
    for traj in range(20):
        desired_goal_state = e.sample_goal()
        states = []
        s = e.reset()
        for step in range(1):
            states.append(s)
            s, _, _, _ = e.step(e.action_space.sample())
            #e.render()
        states = np.stack(states)
        e.get_diagnostics(states[None], desired_goal_state[None])
        logger.dump_tabular()
#     print(np.mean(e.multiworld_env.distances))
#     print(np.mean(e.multiworld_env.times))

if __name__ == "__main__":
    main()
