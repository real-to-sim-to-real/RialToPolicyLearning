"""
This implements Maximum Entropy IRL using dynamic programming. This

Simply call tabular_maxent_irl(env, expert_visitations)
    The expert visitations can be generated via the compute_visitations on an expert q_function (exact),
    or using compute_visitations_demo on demos (approximate)
"""
import numpy as np
from inverse_rl.envs.utils import one_hot_to_flat, flat_to_one_hot
from inverse_rl.envs.tabular.q_iteration import q_iteration, logsumexp, compute_returns
from inverse_rl.utils import TrainingIterator
from inverse_rl.utils.math_utils import gd_momentum_optimizer, adam_optimizer


def get_reward(env, q_fn, ent_wt=1.0, gamma=0.99):
    """ Extract a reward from a q function """
    t_matrix = env.transition_matrix  # S x A x S
    v_fn = logsumexp(q_fn, alpha=ent_wt)
    v_next = t_matrix.dot(v_fn)
    r = q_fn - gamma*v_next
    #adv = q_fn-np.expand_dims(v_fn, axis=1)
    return r


def compute_visitation(env, q_fn, ent_wt=1.0, T=50, discount=1.0):
    pol_probs = get_policy(q_fn, ent_wt=ent_wt)

    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    state_visitation = np.expand_dims(env.initial_state_distribution, axis=1)
    t_matrix = env.transition_matrix  # S x A x S
    sa_visit_t = np.zeros((dim_obs, dim_act, T))

    for i in range(T):
        sa_visit = state_visitation * pol_probs
        sa_visit_t[:,:,i] = sa_visit #(discount**i) * sa_visit
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
    return np.sum(sa_visit_t, axis=2) / float(T)


def compute_vistation_demos(env, demos):
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    counts = np.zeros((dim_obs, dim_act))

    for demo in demos:
        obs = demo['observations']
        act = demo['actions']
        state_ids = one_hot_to_flat(obs)
        T = len(state_ids)
        for t in range(T):
            counts[state_ids[t], act[t]] += 1
    return counts / float(np.sum(counts))


def sample_states(env, q_fn, visitation_probs, n_sample, ent_wt):
    dS, dA = visitation_probs.shape
    samples = np.random.choice(np.arange(dS*dA), size=n_sample, p=visitation_probs.reshape(dS*dA))
    policy = get_policy(q_fn, ent_wt=ent_wt)
    observations = samples // dA
    actions = samples % dA
    a_logprobs = np.log(policy[observations, actions])

    observations_next = []
    for i in range(n_sample):
        t_distr = env.tabular_trans_distr(observations[i], actions[i])
        next_state = flat_to_one_hot(np.random.choice(np.arange(len(t_distr)), p=t_distr), ndim=dS)
        observations_next.append(next_state)
    observations_next = np.array(observations_next)

    return {'observations': flat_to_one_hot(observations, ndim=dS),
            'actions': flat_to_one_hot(actions, ndim=dA),
            'a_logprobs': a_logprobs,
            'observations_next': observations_next}


def tabular_maxent_irl(env, demo_visitations, num_itrs=50, ent_wt=1.0, lr=1e-3, state_only=False,
                       discount=0.99, T=5):
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim

    # Initialize policy and reward function
    reward_fn = np.zeros((dim_obs, dim_act))
    q_rew = np.zeros((dim_obs, dim_act))

    update = adam_optimizer(lr)

    for it in TrainingIterator(num_itrs, heartbeat=1.0):
        q_itrs = 20 if it.itr>5 else 100
        ### compute policy in closed form
        q_rew = q_iteration(env, reward_matrix=reward_fn, ent_wt=ent_wt, warmstart_q=q_rew, K=q_itrs, gamma=discount)

        ### update reward
        # need to count how often the policy will visit a particular (s, a) pair
        pol_visitations = compute_visitation(env, q_rew, ent_wt=ent_wt, T=T, discount=discount)

        grad = -(demo_visitations - pol_visitations)
        it.record('VisitationInfNorm', np.max(np.abs(grad)))
        if state_only:
            grad = np.sum(grad, axis=1, keepdims=True)
        reward_fn = update(reward_fn, grad)

        if it.heartbeat:
            print(it.itr_message())
            print('\t',it.pop_mean('VisitationInfNorm'))
    return reward_fn, q_rew


def inspect_path(path, orig_visitation):
    T, dS = path['observations'].shape
    T, dA = path['actions'].shape
    freq = np.zeros((dS, dA))
    for t in range(T):
        obs = one_hot_to_flat(path['observations'][t])
        act = one_hot_to_flat(path['actions'][t])
        freq[obs, act] += 1
    freq = freq/float(T)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    # test IRL
    np.set_printoptions(suppress=True)
    env = random_env(16, 4, seed=1, terminate=False, t_sparsity=0.8)
    env2 = random_env(16, 4, seed=2, terminate=False, t_sparsity=0.8)
    dS = env.spec.observation_space.flat_dim
    dU = env.spec.action_space.flat_dim
    dO = 8
    ent_wt = 0.5
    discount = 0.7
    obs_matrix = np.random.randn(dS, dO)
    true_q = q_iteration(env, K=150, ent_wt=ent_wt, gamma=discount)
    true_sa_visits = compute_visitation(env, true_q, ent_wt=ent_wt, T=5, discount=discount)
    expert_pol = get_policy(true_q, ent_wt=ent_wt)

    if True:
        learned_rew, learned_q = tabular_maxent_irl(env, true_sa_visits, lr=0.01, num_itrs=1000,
                                                    ent_wt=ent_wt, state_only=True,
                                                    discount=discount)
        #extracted_rew = get_reward(env, learned_q, ent_wt=ent_wt, gamma=discount)
        #new_q = q_iteration(env, K=150, ent_wt=ent_wt, gamma=discount, reward_matrix=extracted_rew)
        learned_pol = get_policy(learned_q, ent_wt=ent_wt)

    else:
        import tensorflow as tf
        from inverse_rl.models.qfunc_irl import DiscreteAdvantageIRL2 as irl_model_cls
        from inverse_rl.models.imitation_learning import GAIL
        from inverse_rl.models.architectures import linear_net
        with tf.Session() as sess:
            irl_model = irl_model_cls(env=env, discount=discount,
                                        discrim_arch=linear_net, 
                                        fitted_value_fn_arch=linear_net,
                                        state_only=True,
                    )
            #irl_model = GAIL(env_spec=env.spec)
            sess.run(tf.global_variables_initializer())
            
            learned_rew, learned_q = tabular_gcl_irl(env, true_sa_visits, irl_model ,lr=0.01, num_itrs=100,
                                                        ent_wt=ent_wt, state_only=False, discount=discount)
            learned_pol = get_policy(learned_q, ent_wt=ent_wt)

    adjusted_rew = learned_rew - np.mean(learned_rew) + np.mean(env.rew_matrix)

    my_ret = compute_returns(env, learned_pol, gamma=discount, ent_wt=ent_wt)
    transfer_q = q_iteration(env2, K=150, ent_wt=ent_wt, gamma=discount, reward_matrix=adjusted_rew)
    transfer_pol = get_policy(transfer_q, ent_wt=ent_wt)
    transfer_ret = compute_returns(env2, transfer_pol, gamma=discount, ent_wt=ent_wt)
    print('Returns Trained Env:', my_ret)
    print('Returns Transfer Env:', transfer_ret)
    import pdb; pdb.set_trace()
    """
    diff_rew = np.abs(env.rew_matrix - adjusted_rew)
    diff_pol = np.abs(expert_pol - learned_pol)
    print('InfNormRew', np.max(diff_rew))
    print('InfNormPol', np.max(diff_pol))
    print('AvdDiffRew', np.mean(diff_rew))
    print('AvgDiffPol', np.mean(diff_pol))
    """
