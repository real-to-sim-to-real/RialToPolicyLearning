import itertools

import gym
from gym.spaces import Box
from gym.spaces import Discrete

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from rllab.misc import logger

from rlutil.envs.env_utils import flat_to_one_hot
import rlutil.log_utils as log_utils
from rlutil.math_utils import np_seed

class DiscreteEnv(gym.Env):
    def __init__(self, transition_matrix, reward, init_state, terminate_on_reward=False,
                 max_timesteps=20, obs_matrix=None):
        super(DiscreteEnv, self).__init__()
        dX, dA, dXX = transition_matrix.shape
        self.num_states = dX
        self.num_actions = dA
        self.transitions = transition_matrix
        self.init_state = init_state
        self.reward = reward
        self.terminate_on_reward = terminate_on_reward
        self.max_timesteps = max_timesteps

        self.obs_matrix = obs_matrix

        if self.obs_matrix is None:
            self.__observation_space = Box(0, 1, shape=(self.num_states,))
        else:
            self.obs_matrix = np.maximum(-1.0, np.minimum(1.0, self.obs_matrix))
            self.__observation_space = Box(-1.0, 1.0, shape=(obs_matrix.shape[0],))

        #max_A = 0
        #for trans in self.transitions:
        #    max_A = max(max_A, len(self.transitions[trans]))
        self.__action_space = Discrete(dA)

    def _wrap_obs(self, state):
        if self.obs_matrix is None:
            return state
        return self.obs_matrix.dot(state)

    def reset(self):
        if isinstance(self.init_state, list):
            self.cur_state = np.random.choice(self.init_state)
        else:
            self.cur_state = self.init_state
        self.__timesteps = 0
        obs = flat_to_one_hot(self.cur_state, ndim=self.num_states)
        return self._wrap_obs(obs)

    def step(self, a):
        transition_probs = self.transitions[self.cur_state, a]
        next_state = np.random.choice(np.arange(self.num_states), p=transition_probs)
        r = self.reward[self.cur_state, a]
        self.cur_state = next_state
        obs = flat_to_one_hot(self.cur_state, ndim=self.num_states)
        self.__timesteps += 1

        done = False
        if (self.terminate_on_reward and r>0) or (self.__timesteps > self.max_timesteps):
            done = True
        return self._wrap_obs(obs), r, done, {}

    def tabular_trans_distr(self, s, a):
        return self.transitions[s, a]

    def reward_fn(self, s, a):
        return self.reward[s, a]

    def log_diagnostics(self, paths):
        #Ntraj = len(paths)
        #acts = np.array([traj['actions'] for traj in paths])
        obs = np.array([np.sum(traj['observations'], axis=0) for traj in paths])

        state_count = np.sum(obs, axis=0)
        #state_count = np.mean(state_count, axis=0)
        state_freq = state_count/float(np.sum(state_count))
        for state in range(self.num_states):
            logger.record_tabular('AvgStateFreq%d'%state, state_freq[state])

    def plot_data(self, data, dirname=None, itr=0, fname='trajs_itr%d'):
        plt.figure()
        ax = plt.gca()
        normalized_values = data
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values/np.max(normalized_values)
        norm_data = normalized_values

        cmap = cm.RdYlBu

        if len(data.shape) == 1:
            ydim = 0
            for x in range(data.shape[0]):
                ax.text(x, 0, '%.1f'%data[x], size='x-small')
                color = cmap(norm_data[x])
                ax.add_patch(Rectangle([x,0],1,1, color=color))
        elif len(data.shape) == 2:
            ydim = data.shape[1]
            for x, y in itertools.product(range(data.shape[0]), range(data.shape[1])):
                iy = data.shape[1]-y-1
                ax.text(x, iy, '%.1f'%data[x, y], size='x-small')
                color = cmap(norm_data[x, y])
                ax.add_patch(Rectangle([x,iy],1,1, color=color))

        ax.set_xticks(np.arange(-1, data.shape[0]+1, 1))
        ax.set_yticks(np.arange(-1, ydim+1, 1))
        plt.grid()

        if dirname is not None:
            log_utils.record_fig(fname%itr, subdir=dirname, rllabdir=True)
        else:
            plt.show()

    def plot_trajs(self, paths, dirname=None, itr=0):
        obs = np.array([np.sum(traj['observations'], axis=0) for traj in paths])
        #state_count = np.sum(obs, axis=1)
        state_count = np.sum(obs, axis=0)
        state_freq = state_count/float(np.sum(state_count))
        self.plot_data(state_freq, dirname=dirname, itr=itr)


    def plot_costs(self, paths, cost_fn, dirname=None, itr=0, policy=None,
                   use_traj_paths=False):
        if not use_traj_paths:
            # iterate through states, and each action - makes sense for non-rnn costs
            obses = []
            acts = []
            for (x, a) in itertools.product(range(self.num_states), range(self.num_actions)):
                obs = flat_to_one_hot(x, ndim=self.num_states)
                act = flat_to_one_hot(a, ndim=self.num_actions)
                obses.append(obs)
                acts.append(act)
            path = {'observations': np.array(obses), 'actions': np.array(acts)}
            if policy is not None:
                if hasattr(policy, 'set_env_infos'):
                    policy.set_env_infos(path.get('env_infos', {}))
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
            paths = [path]

        plots = cost_fn.debug_eval(paths, policy=policy)
        for plot in plots:
            plots[plot] = plots[plot].squeeze()

        for plot in plots:
            data = plots[plot]
            data = np.reshape(data, (self.num_states, self.num_actions))
            self.plot_data(data, dirname=dirname, fname=plot+'_itr%d', itr=itr)

    def transition_matrix(self):
        return self.transitions

    def reward_matrix(self):
        return self.reward

    @property
    def initial_state_distribution(self):
        return flat_to_one_hot(self.init_state, ndim=self.num_states)

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space

    def backup(self, q_fn, gamma):
        v_fn = np.max(q_fn, axis=1)
        new_q = self.rew_matrix + gamma*self.transition_matrix.dot(v_fn)
        return new_q


def random_env(*args, **kwargs):
    return DiscreteEnv(**random_env_register(*args, **kwargs))

def random_env_register(Nstates, Nact, max_timesteps=20,
                        seed=None, terminate=False, t_sparsity=0.75, deterministic=False,
                        dim_obs=-1):
    assert Nstates >= 2
    if seed is None:
        seed = 0
    reward_state=0
    start_state=list(range(1, int(Nstates/2)))
    with np_seed(seed):
        if not deterministic:
            transition_matrix = np.random.rand(Nstates, Nact, Nstates)
            transition_matrix = np.exp(transition_matrix)
            for s in range(Nstates):
                for a in range(Nact):
                    zero_idxs = np.random.randint(0, Nstates, size=int(Nstates*t_sparsity))
                    transition_matrix[s, a, zero_idxs] = 0.0
            transition_matrix = transition_matrix/np.sum(transition_matrix, axis=2, keepdims=True)
        else:
            transition_matrix = np.zeros((Nstates, Nact, Nstates))
            trans_idx = np.random.randint(0, Nstates, size=(Nstates, Nact))
            for state in range(Nstates):
                for act in range(Nact):
                    transition_matrix[state, act, trans_idx[state, act]] = 1.0

        if dim_obs>0:
            obs_matrix = np.random.randn(dim_obs, Nstates)
        else:
            obs_matrix = None

        reward = np.zeros((Nstates, Nact))
        reward[reward_state, :] = 1.0
        #reward = np.random.randn(Nstates,1 ) + reward

        stable_action = seed % Nact #np.random.randint(0, Nact)
        transition_matrix[reward_state, stable_action] = np.zeros(Nstates)
        transition_matrix[reward_state, stable_action, reward_state] = 1
    return {
        'reward': reward,
        'init_state': start_state,
        'terminate_on_reward': terminate,
        'transition_matrix': transition_matrix,
        'max_timesteps': max_timesteps,
        'obs_matrix': obs_matrix,
    }


if __name__ == '__main__':
    env = random_env(5, 2, seed=0)
    print(env.transitions)
    print(env.transitions[0,0])
    print(env.transitions[0,1])
    env.reset()
    for _ in range(100):
        print(env.step(env.action_space.sample()))

