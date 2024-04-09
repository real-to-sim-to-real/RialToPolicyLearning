import os
import sys
import numpy as np
import hashlib
from rlutil.log_utils import DATA_DIR
from rlutil.envs.gridcraft.grid_env import GridEnv 
from rlutil.envs.gridcraft.utils import one_hot_to_flat, flat_to_one_hot

SAVE_DIR = os.path.join(DATA_DIR, 'gridcraft_qvals')


def get_hashname(obj):
    return hex(hash(obj) % ((sys.maxsize+1)*2))[2:]

def hash_env(gridspec, env_args, gamma):
    #d = (get_hashname(gridspec), tuple(env_args.items()), gamma)
    m = hashlib.md5()
    m.update(get_hashname(gridspec).encode('utf-8'))
    m.update(str(tuple(env_args.items())).encode('utf-8'))
    m.update(str(gamma).encode('utf-8'))
    return m.hexdigest()+'.csv'


def dense_tabular_solver(gridspec, env_args, K=50, gamma=0.95, verbose=False, save=True):
    env = GridEnv(gridspec, **env_args)
    # solve with Q-iteration
    dS = len(env.gs)
    dA = env.action_space.n

    # build transition matrix
    transition_matrix = np.zeros((dS, dA, dS))
    for s in range(dS):
        for a in range(dA):
            tprobs = env.get_transitions(s,a)
            for next_s in tprobs:
                transition_matrix[s,a,next_s] = tprobs[next_s]

    # build reward matrix
    rew_matrix = np.zeros((dS, dA, dS))
    for s in range(dS):
        for a in range(dA):
            for ns in range(dS):
                rew_matrix[s,a,ns] = env.rew_fn(env.gs, s, a, ns)
    r_sa = np.sum(transition_matrix * rew_matrix, axis=2)

    q_values = np.zeros((dS, dA))
    prev_diff = 1.0
    for k in range(K):
        v_fn = np.max(q_values, axis=1)  # dO
        new_q = r_sa + gamma * transition_matrix.dot(v_fn)
        diff = np.max(np.abs(new_q - q_values))
        if verbose:
            print(k, 'InfNorm:', diff, 'ContractionFactor:', '%0.4f'%(diff/prev_diff))
        q_values = new_q
        prev_diff = diff

    if save:
        fname = os.path.join(SAVE_DIR, hash_env(gridspec, env_args, gamma=gamma))
        np.savetxt(fname, q_values, delimiter=',')
    return q_values


class QFunc(object):
    def __init__(self, q_arr):
        self._q_arr = q_arr
        self._q_vec = np.reshape(q_arr, [-1])

    def __call__(self, s, a):
        s_idxs = one_hot_to_flat(s)
        return self._q_arr[s_idxs, a]


def load_qvals(gridspec, env_args, gamma=0.95, cache=False):
    fname = os.path.join(SAVE_DIR, hash_env(gridspec, env_args, gamma=gamma))
    if cache and os.path.exists(fname):
        q_arr = np.loadtxt(fname, delimiter=',')
    else:
        print('true_qvalues.py: Running tabular solver...')
        if env_args['max_timesteps'] is not None:
            K = env_args['max_timesteps']
        else:
            K = 500
        q_arr = dense_tabular_solver(gridspec, env_args=env_args, K=K, save=cache)
    return QFunc(q_arr)


def plot_qval(gs, q_values):
    import itertools
    from rlutil.qval_plotter import TabularQValuePlotter
    plotter = TabularQValuePlotter(gs.width, gs.height, text_values=True)
    for i, (x, y, a) in enumerate(itertools.product(range(gs.width), range(gs.height), range(5))):
        plotter.set_value(x, gs.height-y-1, a, q_values[gs.xy_to_idx((x,y)), a])
    plotter.make_plot()
    plotter.show()


if __name__ == "__main__":
    from rlutil.envs.gridcraft.grid_env import GridEnv 
    from rlutil.envs.gridcraft.mazes import *

    #q_values = dense_tabular_solver(MAZE_LAVA, env_args={'teps':0.2}, K=300, verbose=True)
    gs = MAZE_LAVA
    args = {'teps': 0.1}
    q_values = load_qvals(gs, env_args=args)
    print(q_values)



