from gym.envs.registration import register
#import logging


#LOGGER = logging.getLogger(__name__)

_REGISTERED = False
def register_envs():
    from rlutil.envs.gridcraft.mazes import MAZE1, MAZE_LAVA
    from rlutil.envs.gridcraft.grid_spec import REWARD, LAVA
    from rlutil.envs.env_utils import CustomGymEnv
    from rlutil.envs.tabular.simple_env import random_env_register

    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    #LOGGER.info("Registering custom gym environments")
    register(id='GridMaze1-v0', entry_point='rlutil.envs.gridcraft.grid_env:GridEnv',
            kwargs={'gridspec': MAZE1})
    register(id='GridMazeLava-v0', entry_point='rlutil.envs.gridcraft.grid_env:GridEnv',
            kwargs={'gridspec': MAZE_LAVA, 'max_timesteps': 20, 'teps': 0.1})
    register(id='GridMazeLavaEvent-v0', entry_point='rlutil.envs.gridcraft.grid_env:GridEnv',
            kwargs={'gridspec': MAZE_LAVA, 'max_timesteps': 20, 'teps': 0.00,
                    'rew_map': {REWARD:0, LAVA:-100}, 'default_rew': -100})

    register(id='Tabular32x4-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 4, seed=0))
    register(id='Tabular16x4-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(16, 4, seed=0))
    register(id='Tabular8x2-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 2, seed=0))
    register(id='TabularDeterm32x8-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 8, seed=0, deterministic=True))
    register(id='TabularDeterm16x4-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(16, 4, seed=0, deterministic=True))
    register(id='TabularDeterm8x4-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 4, seed=0, deterministic=True))
    register(id='TabularDeterm64x8-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(64, 8, seed=0, deterministic=True))
    register(id='TabularDetermObs32x8-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(32, 8, seed=0, deterministic=True, dim_obs=4))
    register(id='TabularDetermObs8x4-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(8, 4, seed=0, deterministic=True, dim_obs=4))
    register(id='TabularDetermObs64x8-v0', entry_point='rlutil.envs.tabular.simple_env:DiscreteEnv',
             kwargs=random_env_register(64, 8, seed=0, deterministic=True, dim_obs=4))


    #LOGGER.info("Finished registering custom gym environments")
