"""
This exposes helper functions to create envs easily through one interface
TODO(dibyaghosh): This pretty much reimplements gym - should refactor into that

Attributes:
- env_names : a list of all envs that can be accessed through this interace
- image_enabled_env_names : a list of envs that also support image rendering

Functions:
- create_env(env_name, images=False)
- get_goal_threshold(env_name)
- get_oracle_reweighting_fn(env_name, N) 
"""

#from goalsrl.envs.room_env import PointmassGoalEnv
from goalsrl.envs.sawyer_push import SawyerPushGoalEnv
from goalsrl.envs.sawyer_door import SawyerDoorGoalEnv
from goalsrl.envs.env_utils import DiscretizedActionEnv
from goalsrl.envs.ant_env import GoalAntEnv
#from goalsrl.envs.lunarlander import LunarEnv
from rialto.envs.sawyer_push import SawyerPushGoalEnv as SawyerPushGoalEnvObstacle
from rialto.envs.room_env import PointmassGoalEnv 
from rialto.envs.kitchen_env_sequential import KitchenSequentialGoalEnv
#from gcsl.envs.simplified_kitchen_env import KitchenGoalEnv
import numpy as np
env_names = ['pusher', 'kitchen', 'kitchenSeq', 'pointmass_maze', 'pusher_no_obstacle', 'pointmass_rooms', 'pointmass_empty', 'pointmass_wall', 'ant', 'lunar', 'door']
image_enabled_env_names = [] #['pusher', 'pointmass_rooms', 'pointmass_empty', 'pointmass_wall']

_class_names = {
    'pusher_no_obstacle': SawyerPushGoalEnv,
    'pusher': SawyerPushGoalEnvObstacle,
    'door': SawyerDoorGoalEnv,
    'pointmass_rooms': PointmassGoalEnv,
    'pointmass_empty': PointmassGoalEnv,
    'pointmass_wall': PointmassGoalEnv,
    'pointmass_maze':PointmassGoalEnv,
    'ant': GoalAntEnv,
    #'lunar': LunarEnv,
    'kitchenSeq': KitchenSequentialGoalEnv,
    #'kitchen': KitchenGoalEnv
}

_additional_kwargs = {
    'pusher_no_obstacle': dict(fixed_start=True),
    'pusher': dict(fixed_start=True),
    'door': dict(fixed_start=True),
    'pointmass_rooms': dict(room_type='rooms', fixed_start=True),
    'pointmass_empty': dict(room_type='empty', fixed_start=True),
    'pointmass_wall': dict(room_type='wall', fixed_start=True),
    'pointmass_maze': dict(room_type='maze', fixed_start=True),
    'ant': dict(size=4, action_ratio=1, fixed_start=True, fixed_goal=False),
    'lunar': dict(fixed_start=True, fixed_goal=False, frame_skip=2),
    'kitchenSeq': dict(),
}

def create_env(env_name, images=False, fixed_start=True, fixed_goal=False):
    """
    Creates a GoalEnv corresponding to the env passed in

    Arguments:
        env_name (str): The environment to create. Must be in env_names
        images (bool): Whether observation space should be images.
            If True, env_name must be in image_enabled_env_names
    
    Returns:
        env (goalsrl.envs.goal_env.GoalEnv): the environment
    """

    assert env_name in env_names, "Not a valid env name. Choose from %s"%(str(env_names))
    kwargs = _additional_kwargs[env_name].copy()
    if images:
        assert env_name in image_enabled_env_names, "This environment does not have images"
        kwargs['images'] = True
        kwargs['image_kwargs'] = dict(imsize=84)
    
    kwargs['fixed_start'] = fixed_start
    kwargs['fixed_goal'] = fixed_goal
    
    print(env_name, kwargs)
    return _class_names[env_name](**kwargs)

def get_env_params(env_name, images=False):
    base_params = dict(
        eval_freq=2000,
        eval_episodes=50,
    )

    if env_name == 'pusher_no_obstacle':
        env_specific_params = dict(
            max_trajectory_length=50,
            goal_threshold=0.05,
            max_timesteps=1e6,
            oracle_bin_generator=_create_pusher_reweighting,
            eval_freq=10000,
        )
    elif env_name == 'pusher':
        env_specific_params = dict(
            max_trajectory_length=70,
            goal_threshold=0.05,
            max_timesteps=1e6,
            oracle_bin_generator=_create_pusher_reweighting,
            eval_freq=10000,
        )
    elif env_name == 'door':
        env_specific_params = dict(
            max_trajectory_length=50,
            goal_threshold=0.05,
            max_timesteps=1e6,
            oracle_bin_generator=_create_door_reweighting,
            eval_freq=10000,
        )
    elif env_name == 'ant':
        env_specific_params = dict(
            max_trajectory_length=50,
            goal_threshold=0.5,
            max_timesteps=1e6,
            oracle_bin_generator=_create_ant_reweighting,
            action_granularity=2,
            eval_freq=10000,
            eval_episodes=100,
        )
    elif env_name == 'lunar':
        env_specific_params = dict(
            max_trajectory_length=50,
            goal_threshold=0.08,
            max_timesteps=2e5,
            oracle_bin_generator=_create_lunar_reweighting,
        )
    elif 'pointmass' in env_name:
        env_specific_params = dict(
            max_trajectory_length=50,
            goal_threshold=0.08,
            max_timesteps=2e5,
            oracle_bin_generator=_create_pointmass_reweighting,
        )
    elif env_name == 'kitchenSeq':
        env_specific_params = dict(
            max_trajectory_length=70,
            goal_threshold=0.05,
            max_timesteps=1e6,
            oracle_bin_generator=None,
            eval_freq=10000,
        )
    else:
        raise NotImplementedError()
    
    base_params.update(env_specific_params)
    return base_params

def _create_pointmass_reweighting(N):
    def reweighting_fn(goals):
        pos_goals = goals + 0.6
        return ((pos_goals[..., 1] // (1.2 / N )) * N + (pos_goals[..., 0] // (1.2 / N))).astype(int)
    return reweighting_fn, N ** 2

def _create_door_reweighting(N):
    def reweighting_fn(goals):
        pos_goals = np.clip(goals + 0.1, 0, 1.2)
        return (pos_goals[..., 3] // (1.2 / N)).astype(int)
    return reweighting_fn, N

def _create_lunar_reweighting(N):
    def reweighting_fn(goals):
        pos_goals = np.clip(goals[..., 0] + 0.6, 0, 1.2)
        return (pos_goals[..., 0] // (1.2 / N)).astype(int)
    return reweighting_fn, N

def _create_ant_reweighting(N):
    def reweighting_fn(goals):
        pos_goals = goals + 2
        pos_goals = np.clip(pos_goals, 0.0, 4.0)
        return ((pos_goals[..., 1] // (4.0 / N )) * N + (pos_goals[..., 0] // (4.0 / N))).astype(int)
    return reweighting_fn, N ** 2

def _create_pusher_reweighting(N):
    def reweighting_fn(goals):
        pos_goals = goals.copy()
        pos_goals[..., 0] += 0.2
        pos_goals[..., 1] -= 0.5
        pos_goals[..., 2] += 0.2
        pos_goals[..., 3] -= 0.5
        clip = lambda x: np.clip(x, 0, N-1)

        return (
            clip(pos_goals[..., 0] // (0.4 / N)) * (N**0) + \
            clip(pos_goals[..., 1] // (0.2 / N)) * (N**1) + \
            clip(pos_goals[..., 2] // (0.4 / N)) * (N**2) + \
            clip(pos_goals[..., 3] // (0.2 / N)) * (N**3) 
        ).astype(int)
    return reweighting_fn, N ** 4