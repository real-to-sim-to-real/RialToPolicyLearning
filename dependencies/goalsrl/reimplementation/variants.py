from goalsrl.reimplementation import buffer, networks
from goalsrl.envs.env_utils import DiscretizedActionEnv
from gym.spaces import Box, Discrete
import numpy as np

# Main Function
all_variants = [
    'default', 'no_reweight', 'no_horizon',
    'oracle_reweight', 'rwr', 'inverse_model',
    'no_iteration', 'on_policy',
]

def get_params(env, env_params, variant):
    """
    Returns everything needed for gcsl to run

    Arguments:
        env - The goalsrl GoalEnv to train
        env_params (dict) - A list of environment and algorithm parameters
        variant (str) - The algorithm variant to run (choose from all_variants)

    Returns:
        env - (potentially different from env passed in!) use this env
        policy - The policy to train
        replay_buffer - The replay buffer in which everything will be stored
        gcsl_kwargs - Additional kwargs to use for GCSL
    """

    assert variant in all_variants, "%s not found in %s"%(variant, str(all_variants))

    if variant == 'default':
        raise NotImplementedError()
    elif variant == 'no_reweight':
        return get_noreweight_params(env, env_params)
    elif variant == 'no_horizon':
        return get_nohorizon_params(env, env_params)
    elif variant == 'oracle_reweight':
        return get_oracle_reweight_params(env, env_params)
    elif variant == 'rwr':
        return get_rwr_params(env, env_params)
    elif variant == 'inverse_model':
        return get_inversemodel_params(env, env_params)
    elif variant == 'no_iteration':
        return get_notiterated_params(env, env_params)
    elif variant == 'on_policy':
        return get_onpolicy_params(env, env_params)
    else:
        raise NotImplementedError()

# UTILITY FUNCTIONS

def is_image_env(env):
    return len(env.observation_space.shape) > 1

def get_image_resolution(env):
    assert is_image_env(env), "Only works with images"
    return env.observation_space.shape[1]

def get_horizon(env_params):
    return env_params.get('max_trajectory_length', 50)

def discretize_environment(env, env_params):
    if isinstance(env.action_space, Discrete):
        return env
    granularity = env_params.get('action_granularity', 3)
    env_discretized = DiscretizedActionEnv(env, granularity=granularity)
    return env_discretized

# DEFAULT POLICIES
def default_policy(env, max_horizon=None):
    if isinstance(env.action_space, Discrete):
        if not is_image_env(env):
            return networks.DiscreteStochasticGoalPolicy(
                env,
                state_embedding=None,
                goal_embedding=None,
                layers=[400, 300], # TD3-size
                max_horizon=max_horizon,
                freeze_embeddings=True,
                add_extra_conditioning=False,
            )
        else:
            imsize = get_image_resolution(env)
            return networks.DiscreteStochasticGoalPolicy(
                env,
                state_embedding=networks.CNNHead(image_size=imsize, spatial_softmax=True),
                goal_embedding=networks.CNNHead(image_size=imsize, spatial_softmax=True),
                layers=[400, 300],
                max_horizon=max_horizon,
                freeze_embeddings=False,
                add_extra_conditioning=False,
            )

    elif isinstance(env.action_space, Box):
        if is_image_env(env):
            raise NotImplementedError()
        else:
            raise NotImplementedError()

def default_horizon_policy(env, env_params):
    max_horizon = get_horizon(env_params)
    return default_policy(env, max_horizon=max_horizon)

def default_markov_policy(env, env_params):
    return default_policy(env, max_horizon=None)

# DEFAULT BUFFER STUFF
def default_buffer_kwargs(env, env_params):
    max_trajectory_length = get_horizon(env_params)
    max_buffer_size = 2000 if is_image_env(env) else 20000
    return dict(env=env, max_trajectory_length=max_trajectory_length,  buffer_size=max_buffer_size)

def default_oracle_buffer_kwargs(env, env_params):
    params = default_buffer_kwargs(env, env_params)

    oracle_kwargs = env_params.get('oracle_kwargs', dict())
    granularity = oracle_kwargs.get('granularity', 10)
    oracle_bin_generator = env_params.get('oracle_bin_generator')
    oracle_reweight_fn, oracle_n_bins = oracle_bin_generator(granularity)

    params.update(
        dict(
            reweight_fn=oracle_reweight_fn,
            n_reweight_bins=oracle_n_bins,
            use_internal_goals=True,
        )
    )
    return params
# DEFAULT ALGO PARAMS

def default_gcsl_params(env, env_params):
    image_env = is_image_env(env)
    return dict(
        max_path_length=env_params.get('max_trajectory_length', 50),
        goal_threshold=env_params.get('goal_threshold', 0.05),
        start_timesteps=10000,
        start_policy_timesteps=1000,
        eval_freq=env_params.get('eval_freq', 2000),
        eval_episodes=env_params.get('eval_episodes', 50),
        save_every_iteration=False,
        max_timesteps=env_params.get('max_timesteps', 1e6),
        expl_noise=0.2,
        batch_size=256,
        n_accumulations=1,
        policy_updates_per_step=1,
        reset_policy_freq=float('inf'),
        train_policy_freq=None,
        lr=5e-4,
    )

############
# VANILLA
############

def get_noreweight_params(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_horizon_policy(env, env_params)
    replay_buffer = buffer.ReplayBuffer(**default_buffer_kwargs(env, env_params))
    gcsl_kwargs = default_gcsl_params(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs

####################
# ORACLE REWEIGHTING
####################

def get_oracle_reweight_params(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_horizon_policy(env, env_params)



    replay_buffer = buffer.BinWeightedReplayBuffer(
        **default_oracle_buffer_kwargs(env, env_params)
    )
    gcsl_kwargs = default_gcsl_params(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs


##############
# NO HORIZON
##############

def get_nohorizon_params(env, env_params):
    env, _, replay_buffer, gcsl_kwargs = get_oracle_reweight_params(env, env_params)
    policy = default_markov_policy(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs


##########
# RWR
##########

class DiscountedBuffer(buffer.BinWeightedReplayBuffer):
    def __init__(self, discount=0.99, **kwargs):
        super().__init__(**kwargs)
        self.discount = discount

    def _get_batch(self, traj_idxs, time_state_idxs, time_goal_idxs):
        observations, actions, goals, lengths, horizons, weights = super()._get_batch(traj_idxs, time_state_idxs, time_goal_idxs)
        decayed_weights = self.discount ** lengths
        return observations, actions, goals, lengths, horizons, decayed_weights

def get_rwr_params(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_markov_policy(env, env_params)

    rwr_kwargs = env_params.get('rwr_kwargs', dict())
    discount = rwr_kwargs.get('discount', 1 - 1 / get_horizon(env_params))

    buffer_params = default_oracle_buffer_kwargs(env, env_params)
    buffer_params.update(dict(discount=discount))
    replay_buffer = DiscountedBuffer(**buffer_params)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs

###############
# INVERSE MODEL
###############

class InverseModelBuffer(buffer.BinWeightedReplayBuffer):
    def __init__(self, env, max_trajectory_length, buffer_size, lookahead=5, **kwargs):
        super().__init__(env, max_trajectory_length, buffer_size, imagine_horizon=False, **kwargs)
        self.lookahead = lookahead

    def _sample_indices(self, batch_size):
        traj_idxs, _, time_goal_idxs = super()._sample_indices(batch_size)
        sub = np.random.randint(self.lookahead, size=time_goal_idxs.shape)
        time_state_idxs = np.clip(time_goal_idxs - sub, 0, self.max_trajectory_length).astype(int)
        return traj_idxs, time_state_idxs, time_goal_idxs

def get_inversemodel_params(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_markov_policy(env, env_params)

    inversemodel_kwargs = env_params.get('inversemodel_kwargs', dict())
    lookahead = inversemodel_kwargs.get('lookahead', 1)

    replay_buffer = InverseModelBuffer(lookahead=lookahead, **default_oracle_buffer_kwargs(env, env_params))
    gcsl_kwargs = default_gcsl_params(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs

################
# NOT ITERATED
################

def get_notiterated_params(env, env_params):
    env, policy, replay_buffer, gcsl_kwargs = get_oracle_reweight_params(env, env_params)
    gcsl_kwargs['start_timesteps'] = gcsl_kwargs['max_timesteps']
    return env, policy, replay_buffer, gcsl_kwargs

################################
# Only New Data Every Iteration
################################

def get_onpolicy_params(env, env_params):
    env = discretize_environment(env, env_params)
    policy = default_horizon_policy(env, env_params)
    buffer_kwargs = default_oracle_buffer_kwargs(env, env_params)
    buffer_kwargs['buffer_size'] = 200
    replay_buffer = buffer.BinWeightedReplayBuffer(**buffer_kwargs)
    gcsl_kwargs = default_gcsl_params(env, env_params)
    return env, policy, replay_buffer, gcsl_kwargs
