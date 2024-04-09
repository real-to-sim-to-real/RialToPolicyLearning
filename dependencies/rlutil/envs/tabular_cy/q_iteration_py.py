"""Implementations of hard and soft q-iteration for TabularEnv environments."""
import numpy as np
import scipy.misc
from scipy.misc import logsumexp as lse


def compute_value_function(q_values, ent_wt=0.0):
    if ent_wt > 0:
        # soft max
        v_fn = ent_wt * lse((1.0 / ent_wt) * q_values, axis=1, keepdims=False)
    else:
        # hard max
        v_fn = np.max(q_values, axis=1)
    return v_fn


def q_iteration_dense(transition_matrix,
                      reward_matrix,
                      warmstart_q=None,
                      num_itrs=100,
                      ent_wt=0.0,
                      discount=0.99,
                      atol=1e-8):
    """Computes optimal q-values using dense q-iteration.

    Args:
      transition_matrix:  A dS x dA x dS transition matrix
      reward_matrix: A dS x dA x dS reward matrix
      warmstart_q: A dS x dA array of initial q-values.
      num_itrs: Number of iterations to run.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.
      atol: Absolute error tolerance for early stopping.

    Returns:
      A dS x dA array of Q-values
    """
    ds, da, _ = transition_matrix.shape

    q_vals = warmstart_q
    if warmstart_q is None:
        q_vals = np.zeros((ds, da), dtype=np.float64)

    transition_rewards = np.sum(transition_matrix * reward_matrix, axis=2)

    for _ in range(num_itrs):
        v_fn = compute_value_function(q_vals, ent_wt)

        new_q = transition_rewards + \
            discount * transition_matrix.dot(v_fn)
        diff = np.max(np.abs(new_q - q_vals))
        q_vals = new_q
        if diff < atol:
            break
    return q_vals


def q_iteration_sparse(tabular_env,
                       reward_fn=None,
                       warmstart_q=None,
                       num_itrs=100,
                       ent_wt=0.0,
                       discount=0.99):
    """Computes optimal q-values using sparse q-iteration.

    Args:
      tabular_env: A TabularEnv environment.
      reward_fn: A scalar-valued reward function f(s, a, ns) -> reward
      num_itrs: Number of iterations to run.
      warmstart_q: A dS x dA array of initial q-values.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.

    Returns:
      A dS x dA array of Q-values
    """
    ds = tabular_env.num_states
    da = tabular_env.num_actions

    if reward_fn is None:
        reward_fn = tabular_env.reward

    q_vals = warmstart_q
    if warmstart_q is None:
        q_vals = np.zeros((ds, da), dtype=np.float64)

    for _ in range(num_itrs):
        v_fn = compute_value_function(q_vals, ent_wt)
        new_q = np.zeros((ds, da))

        for s in range(ds):
            for a in range(da):
                transitions = tabular_env.transitions(s, a)
                #TODO(justinjfu): use next state???
                r_sa = reward_fn(s, a, 0)
                for ns in transitions:
                    new_q[s, a] += transitions[ns] * \
                        (r_sa + discount * v_fn[ns])
        q_vals = new_q
    return q_vals

