# distutils: language=c++
"""Cython implementation of Q-iteration."""
import cython

import numpy as np
import scipy.misc
from scipy.misc import logsumexp as lse

from cython.operator cimport dereference, preincrement
from libcpp.map cimport map
from libc.math cimport fmax, fabs, exp, log

from rlutil.envs.tabular_cy cimport tabular_env
from rlutil.envs.tabular_cy.q_iteration_py import compute_value_function as compute_value_function_py


@cython.boundscheck(False)
@cython.cdivision(True)
cdef compute_value_function(double[:, :] q_values, double[:] values, int ds, int da, double ent_wt):
    r"""Computes the value function by maxing over the q-values.

    Args:
      q_values: A dS x dA array of q-values.
      values: A dS array where the result will be stored
      ds: Number of states
      da: Number of actions
      ent_wt: Entropy weight. Default 0.
    """
    cdef int s, a
    cdef double max_val, total

    if ent_wt > 0:
        for s in range(ds):
            max_val = q_values[s, 0]
            for a in range(da):
                max_val = fmax(max_val, q_values[s,a])

            total = 0
            for a in range(da):
                total += exp((q_values[s, a] - max_val)/ent_wt)
            values[s] = max_val + ent_wt * log(total)
    else:
        for s in range(ds):
            max_val = q_values[s, 0]
            for a in range(da):
                max_val = fmax(max_val, q_values[s,a])
            values[s] = max_val

@cython.boundscheck(False)
@cython.cdivision(True)
cdef compute_value_function_evaluation(double[:, :] q_values, double[:] values, double[:,:] policy, int ds, int da, double ent_wt):
    r"""Computes the (policy evaluation) value function 

    Args:
      q_values: A dS x dA array of q-values.
      values: A dS array where the result will be stored
      ds: Number of states
      da: Number of actions
      ent_wt: Entropy weight. Default 0.
    """
    cdef int s, a
    cdef double max_val, total

    if ent_wt > 0:
        for s in range(ds):
            total = 0
            for a in range(da):
                total += policy[s,a] * (q_values[s, a] - log(policy[s,a] + 1e-8))
            values[s] = total
    else:
        for s in range(ds):
            exp_val = 0
            for a in range(da):
                exp_val += policy[s, a] * q_values[s,a]
            values[s] = exp_val


@cython.boundscheck(False)
@cython.cdivision(True)
cdef compute_value_function_masked(double[:, :] q_values, double[:] values, int ds, int da, double ent_wt, double[:,:] mask):
    r"""Computes the value function by maxing over the q-values.

    Args:
      q_values: A dS x dA array of q-values.
      values: A dS array where the result will be stored
      ds: Number of states
      da: Number of actions
      ent_wt: Entropy weight. Default 0.
    """
    cdef int s, a
    cdef double max_val, total

    if ent_wt > 0:
        for s in range(ds):
            max_val = q_values[s, 0]
            for a in range(da):
                if mask[s,a] > 0:
                    max_val = fmax(max_val, q_values[s,a])

            total = 0
            for a in range(da):
                if mask[s,a] > 0:
                    total += exp((q_values[s, a] - max_val)/ent_wt)
            values[s] = max_val + ent_wt * log(total)
    else:
        for s in range(ds):
            max_val = q_values[s, 0]
            for a in range(da):
                if mask[s,a] > 0:
                    max_val = fmax(max_val, q_values[s,a])
            values[s] = max_val


@cython.boundscheck(False)
cdef double max_abs_error(double[:, :] q1, double[:, :] q2, int ds, int da):
    """Compute max absolute error between two q values for early stopping."""
    cdef double max_error
    max_error = 0.0
    for s in range(ds):
        for a in range(da):
            max_error = fmax(max_error, fabs(q1[s,a]-q2[s,a]))
    return max_error


@cython.boundscheck(False)
cpdef q_iteration_sparse_python(tabular_env,
                                reward_fn=None,
                                warmstart_q=None,
                                int num_itrs=100,
                                double ent_wt=0.0,
                                double discount=0.99,
                                double atol=1e-8):
    """Computes q-values using sparse q-iteration.

    Args:
      tabular_env: A python TabularEnv environment.
      reward_fn: A scalar-valued reward function f(s, a, ns) -> reward
      warmstart_q: A dS x dA array of initial q-values.
      num_itrs: Number of iterations to run.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.
      atol: Absolute error tolerance for early stopping.

    Returns:
      A dS x dA array of Q-values
    """
    cdef int ds, da, s, a, i, ns_idx, ns
    ds = tabular_env.num_states
    da = tabular_env.num_actions

    if reward_fn is None:
        reward_fn = tabular_env.reward

    q_values_np = np.zeros((ds, da), dtype=np.float64)
    if warmstart_q is not None:
        q_values_np[:, :] = warmstart_q
    cdef double[:, :] q_values = q_values_np

    new_q_values_np = np.zeros((ds, da), dtype=np.float64)
    cdef double[:, :] new_q_values = new_q_values_np

    r_sa_np = np.zeros((ds, da), dtype=np.float64)
    cdef double[:, :] r_sa = r_sa_np
    for s in range(ds):
        for a in range(da):
            r_sa[s, a] = reward_fn(s, a, 0)

    v_fn_np = np.zeros((ds), dtype=np.float64)
    cdef double[:] v_fn = v_fn_np

    for i in range(num_itrs):
        compute_value_function(q_values, v_fn, ds, da, ent_wt)

        new_q_values[:, :] = 0.0
        for s in range(ds):
            for a in range(da):
                transitions_py = tabular_env.transitions(s, a)
                for ns in transitions_py:
                    new_q_values[s, a] += transitions_py[ns] * \
                        (r_sa[s, a] + discount * v_fn[ns])

        if atol > 0:
            diff = max_abs_error(new_q_values, q_values, ds, da)
            if diff < atol:
                break
        q_values[:, :] = new_q_values[:, :]
    return q_values_np


@cython.boundscheck(False)
cpdef softq_iteration(tabular_env.TabularEnv tabular_env,
                         warmstart_q=None,
                         mask=None,
                         int num_itrs=100,
                         double ent_wt=0.0,
                         double discount=0.99,
                         double atol=1e-8):
    """Computes q-values using sparse q-iteration.

    Args:
      tabular_env: A cython TabularEnv environment.
      warmstart_q: A dS x dA array of initial q-values.
      num_itrs: Number of iterations to run.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.
      atol: Absolute error tolerance for early stopping.
        If atol < 0, this will always run for num_itrs iterations.

    Returns:
      A dS x dA array of Q-values
    """
    cdef int ds, da, s, a, i, ns_idx, ns
    ds = tabular_env.num_states
    da = tabular_env.num_actions

    q_values_np = np.zeros((ds, da), dtype=np.float64)
    if warmstart_q is not None:
        q_values_np[:, :] = warmstart_q
    cdef double[:, :] q_values = q_values_np

    
    cdef double[:, :] mask_;
    if mask is not None:
        mask_ = mask

    new_q_values_np = np.zeros((ds, da), dtype=np.float64)
    cdef double[:, :] new_q_values = new_q_values_np

    v_fn_np = np.zeros((ds), dtype=np.float64)
    cdef double[:] v_fn = v_fn_np

    for i in range(num_itrs):
        if mask is not None:
            compute_value_function_masked(q_values, v_fn, ds, da, ent_wt, mask=mask_)
        else:
            compute_value_function(q_values, v_fn, ds, da, ent_wt)

        new_q_values[:, :] = 0.0
        for s in range(ds):
            for a in range(da):
                transitions = tabular_env.transitions_cy(s, a)
                transitions_end = transitions.end()
                transitions_it = transitions.begin()
                reward = tabular_env.reward(s, a, 0)
                while transitions_it != transitions_end:
                    ns = dereference(transitions_it).first
                    p = dereference(transitions_it).second
                    new_q_values[s, a] += p * (reward + discount * v_fn[ns])
                    preincrement(transitions_it)
        if atol > 0:
            diff = max_abs_error(new_q_values, q_values, ds, da)
            if diff < atol:
                break
        q_values[:, :] = new_q_values[:, :]
    return q_values_np


@cython.boundscheck(False)
cpdef softq_evaluation(tabular_env.TabularEnv tabular_env,
                       eval_qvalues,
                       warmstart_q=None,
                       int num_itrs=100,
                       double ent_wt=0.0,
                       double discount=0.99,
                       double atol=1e-8):
    """Computes q-values using sparse q-iteration.

    Args:
      tabular_env: A cython TabularEnv environment.
      warmstart_q: A dS x dA array of initial q-values.
      num_itrs: Number of iterations to run.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.
      atol: Absolute error tolerance for early stopping.
        If atol < 0, this will always run for num_itrs iterations.

    Returns:
      A dS x dA array of Q-values
    """
    cdef int ds, da, s, a, i, ns_idx, ns
    ds = tabular_env.num_states
    da = tabular_env.num_actions

    q_values_np = np.zeros((ds, da), dtype=np.float64)
    if warmstart_q is not None:
        q_values_np[:, :] = warmstart_q
    cdef double[:, :] q_values = q_values_np

    new_q_values_np = np.zeros((ds, da), dtype=np.float64)
    cdef double[:, :] new_q_values = new_q_values_np

    v_fn_np = np.zeros((ds), dtype=np.float64)
    cdef double[:] v_fn = v_fn_np

    if ent_wt > 0:
        compute_value_function(eval_qvalues, v_fn_np, ds, da, ent_wt)
        policy_np = np.exp(eval_qvalues - np.expand_dims(v_fn_np, axis=-1))
        policy_np = policy_np / np.sum(policy_np, axis=1, keepdims=True)
    else:
        policy_np = np.zeros_like(q_values_np)
        policy_np = np.eye(da)[np.argmax(eval_qvalues, axis=1)] * eval_qvalues
        policy_np[policy_np>0] = 1.0
    cdef double[:,:] policy = policy_np

    for i in range(num_itrs):
        compute_value_function_evaluation(q_values, v_fn, policy, ds, da, ent_wt)

        new_q_values[:, :] = 0.0
        for s in range(ds):
            for a in range(da):
                transitions = tabular_env.transitions_cy(s, a)
                transitions_end = transitions.end()
                transitions_it = transitions.begin()
                reward = tabular_env.reward(s, a, 0)
                while transitions_it != transitions_end:
                    ns = dereference(transitions_it).first
                    p = dereference(transitions_it).second
                    new_q_values[s, a] += p * (reward + discount * v_fn[ns])
                    preincrement(transitions_it)
        if atol > 0:
            diff = max_abs_error(new_q_values, q_values, ds, da)
            if diff < atol:
                break
        q_values[:, :] = new_q_values[:, :]

    compute_value_function_evaluation(q_values, v_fn, policy, ds, da, ent_wt)
    exp_returns = 0
    for state in tabular_env.initial_state_distribution:
        exp_returns += v_fn[state]
    return exp_returns, q_values_np


@cython.boundscheck(False)
cpdef softq_iteration_custom_reward(tabular_env.TabularEnv tabular_env,
                         reward,
                         warmstart_q=None,
                         int num_itrs=100,
                         double ent_wt=0.0,
                         double discount=0.99,
                         double atol=1e-8):
    """Computes q-values using sparse q-iteration.

    Args:
      tabular_env: A cython TabularEnv environment.
      reward: A dS x dA reward matrix
      warmstart_q: A dS x dA array of initial q-values.
      num_itrs: Number of iterations to run.
      ent_wt: Entropy weight. Default 0.
      discount: Discount factor.
      atol: Absolute error tolerance for early stopping.
        If atol < 0, this will always run for num_itrs iterations.

    Returns:
      A dS x dA array of Q-values
    """
    cdef int ds, da, s, a, i, ns_idx, ns
    ds = tabular_env.num_states
    da = tabular_env.num_actions

    q_values_np = np.zeros((ds, da), dtype=np.float64)
    if warmstart_q is not None:
        q_values_np[:, :] = warmstart_q
    cdef double[:, :] q_values = q_values_np

    cdef double[:,:] custom_reward = reward

    new_q_values_np = np.zeros((ds, da), dtype=np.float64)
    cdef double[:, :] new_q_values = new_q_values_np

    v_fn_np = np.zeros((ds), dtype=np.float64)
    cdef double[:] v_fn = v_fn_np

    for i in range(num_itrs):
        compute_value_function(q_values, v_fn, ds, da, ent_wt)

        new_q_values[:, :] = 0.0
        for s in range(ds):
            for a in range(da):
                transitions = tabular_env.transitions_cy(s, a)
                transitions_end = transitions.end()
                transitions_it = transitions.begin()
                r = custom_reward[s, a]
                while transitions_it != transitions_end:
                    ns = dereference(transitions_it).first
                    p = dereference(transitions_it).second
                    new_q_values[s, a] += p * (r + discount * v_fn[ns])
                    preincrement(transitions_it)
        if atol > 0:
            diff = max_abs_error(new_q_values, q_values, ds, da)
            if diff < atol:
                break
        q_values[:, :] = new_q_values[:, :]
    return q_values_np


#@cython.cdivision(True)
cpdef get_policy(q_fn, double ent_wt=1.0):
    """Return a policy by normalizing a Q-function."""
    cdef double inverse_ent = 1.0/ent_wt
    value_fn = ent_wt * lse(inverse_ent * q_fn, axis=1)
    adv_rew = q_fn - np.expand_dims(value_fn, axis=1)
    pol_probs = np.exp(inverse_ent * adv_rew)
    return pol_probs
