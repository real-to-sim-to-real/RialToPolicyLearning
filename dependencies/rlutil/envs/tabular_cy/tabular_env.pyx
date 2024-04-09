# distutils: language=c++
"""Base class for cython-based tabular envs.

Subclasses should implement the transitions_cy,  reward methods.
An example environment is provided in CliffwalkEnv
"""
import gym
import gym.spaces
import numpy as np
import cython
from rlutil.envs.tabular_cy.tabular_env cimport TimeStep, PendulumState
from rlutil.math_utils import np_seed

from libc.math cimport fmin, fmax, sin, cos, pi, floor
from libcpp.map cimport map, pair
from libc.stdlib cimport rand
cdef extern from "limits.h":
    int INT_MAX
from cython.operator cimport dereference, preincrement


@cython.cdivision(True)
cdef inline int sample_int(map[int, double] transitions):
    cdef float randnum = rand() / float(INT_MAX)
    cdef float total = 0
    transitions_end = transitions.end()
    transitions_it = transitions.begin()
    while transitions_it != transitions_end:
        ns = dereference(transitions_it).first
        p = dereference(transitions_it).second
        
        if (p+total) >= randnum:
            return ns
        total += p
        preincrement(transitions_it)


cdef class TabularEnv(object):
    """Base class for tabular environments.

    States and actions are represented as integers ranging from
    [0,  self.num_states) or [0, self.num_actions), respectively.

    Args:
      num_states: Size of the state space.
      num_actions: Size of the action space.
      initial_state_distribution: A dictionary from states to
        probabilities representing the initial state distribution.
    """

    def __init__(self,
                 int num_states,
                 int num_actions,
                 dict initial_state_distribution):
        self._state = -1
        self.observation_space = gym.spaces.Discrete(num_states)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.num_states = num_states
        self.num_actions = num_actions
        self.initial_state_distribution = initial_state_distribution

    cpdef transitions(self, int state, int action):
        """Computes transition probabilities p(ns|s,a).

        Args:
          state:
          action:

        Returns:
          A python dict from {next state: probability}.
          (Omitted states have probability 0)
        """
        return dict(self.transitions_cy(state, action))

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        self._transition_map.insert(pair[int, double](state, 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        """Return the reward

        Args:
          state:
          action: 
          next_state: 
        """
        return 0.0

    cpdef observation(self, int state):
        """Computes observation for a given state.

        Args:
          state: 

        Returns:
          observation: Agent's observation of state, conforming with observation_space
        """
        return state

    cpdef step(self, int action):
        """Simulates the environment by one timestep.

        Args:
          action: Action to take

        Returns:
          observation: Next observation
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        infos = {'state': self.get_state()}
        ts = self.step_state(action)
        nobs = self.observation(ts.state)
        return nobs, ts.reward, ts.done, infos

    @cython.infer_types(True)
    cpdef TimeStep step_state(self, int action):
        """Simulates the environment by one timestep, returning the state id
        instead of the observation.

        Args:
          action: Action taken by the agent.

        Returns:
          state: Next state
          reward: Reward incurred by agent
          done: A boolean indicating the end of an episode
          info: A debug info dictionary.
        """
        cdef int next_state
        transitions = self.transitions_cy(self._state, action)
        #next_state = np.random.choice(
        #    list(transitions.keys()), p=list(transitions.values()))
        next_state = sample_int(transitions)
        reward = self.reward(self.get_state(), action, next_state)
        self.set_state(next_state)
        return TimeStep(next_state, reward, False)

    cpdef reset(self):
        """Resets the state of the environment and returns an initial observation.

        Returns:
          observation (object): The agent's initial observation.
        """
        initial_state = self.reset_state()
        return self.observation(initial_state)

    cpdef int reset_state(self):
        """Resets the state of the environment and returns an initial state.

        Returns:
          state: The agent's initial state
        """
        initial_states = list(self.initial_state_distribution.keys())
        initial_probs = list(self.initial_state_distribution.values())
        initial_state = np.random.choice(initial_states, p=initial_probs)
        self.set_state(initial_state)
        return initial_state

    @cython.boundscheck(False)
    cpdef transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corrsponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        cdef int next_s
        ds = self.num_states
        da = self.num_actions
        transition_matrix_np = np.zeros((ds, da, ds))
        cdef double[:, :, :] transition_matrix = transition_matrix_np
        for s in range(ds):
            for a in range(da):
                transitions = self.transitions_cy(s, a)
                transitions_end = transitions.end()
                transitions_it = transitions.begin()
                while transitions_it != transitions_end:
                    next_s = dereference(transitions_it).first
                    prob = dereference(transitions_it).second
                    transition_matrix[s, a, next_s] = prob
                    preincrement(transitions_it)
        return transition_matrix_np

    @cython.boundscheck(False)
    cpdef reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA x dS numpy array where the entry reward_matrix[s, a, ns]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        rew_matrix_np = np.zeros((ds, da, ds))
        cdef double[:, :, :] rew_matrix = rew_matrix_np
        for s in range(ds):
            for a in range(da):
                for ns in range(ds):
                    rew_matrix[s, a, ns] = self.reward(s, a, ns)
        return rew_matrix_np

    cpdef set_state(self, int state):
        """Set the agent's internal state."""
        self._state = state

    cpdef int get_state(self):
        """Return the agent's internal state."""
        return self._state

    cpdef render(self):
        """Render the current state of the environment."""
        print(self.get_state())     


cdef class CliffwalkEnv(TabularEnv):
    """An example env where an agent can move along a sequence of states. There is
    a chance that the agent may jump back to the initial state.

    Action 0 moves the agent back to start, and action 1 to the next state.
    The agent only receives reward in the final state and is forced to move back to the start.

    Args:
      num_states: Number of states 
      transition_noise: A float in [0, 1] representing the chance that the
        agent will be transported to the start state.
    """

    def __init__(self, int num_states=3, double transition_noise=0.0):
        super(CliffwalkEnv, self).__init__(num_states, 2, {0: 1.0})
        self.transition_noise = transition_noise

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        if action == 0:
            self._transition_map.insert(pair[int, double](0, 1.0))
        else:
            if state == self.num_states-1:
                self._transition_map.insert(pair[int, double](0, 1.0))
            else:
                self._transition_map.insert(
                    pair[int, double](0, self.transition_noise))
                self._transition_map.insert(pair[int, double](state + 1, 1.0 - self.transition_noise))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        if state == self.num_states - 1 and action == 1:
            return 1.0
        else:
            return 0.0


cdef class RandomTabularEnv(TabularEnv):
    def __init__(self, int num_states=3, int num_actions=2, int transitions_per_action=4, int seed=0,
                bint self_loop=0):
        super(RandomTabularEnv, self).__init__(num_states, num_actions, {0: 1.0})

        with np_seed(seed):
            rewards = np.zeros((num_states, num_actions))
            reward_state = np.random.randint(1, num_states)
            rewards[reward_state, :] = 1.0
            self._reward_matrix = rewards

            transition_matrix = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
            scores = np.random.rand(num_states, num_actions, num_states).astype(np.float64)
            scores[:, :, reward_state] *= 0.999  # reduce chance of link to goal

            for s in range(num_states):
                for a in range(num_actions):
                    top_states = np.argsort(scores[s, a, :])[-transitions_per_action:]
                    for ns in top_states:
                        transition_matrix[s, a, ns] = 1.0/float(transitions_per_action)
                if self_loop:
                    for ns in range(num_states):
                        transition_matrix[s, 0, ns] = 0.0
                    transition_matrix[s, 0, s] = 1.0
            transition_matrix = transition_matrix/np.sum(transition_matrix, axis=2, keepdims=True)
            self._transition_matrix = transition_matrix

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        cdef int ns
        cdef double prob
        for ns in range(self.num_states):
            prob = self._transition_matrix[state, action, ns]
            if prob > 0:
                self._transition_map.insert(pair[int, double](ns, prob))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        return self._reward_matrix[state, action]


cdef class InvertedPendulum(TabularEnv):
    """ 
    Dynamics and reward are based on OpenAI gym's implementation of Pendulum-v0
    """

    def __init__(self, int state_discretization=64, int action_discretization=5):
        self._state_disc = state_discretization
        self._action_disc = action_discretization
        self.max_vel = 4.
        self.max_torque = 3.

        self.action_map = np.linspace(-self.max_torque, self.max_torque, num=action_discretization)
        self.state_map = np.linspace(-pi, pi, num=state_discretization)
        self._state_min = -pi 
        self._state_step = (2*pi) / state_discretization
        self.vel_map = np.linspace(-self.max_vel, self.max_vel, num=state_discretization)
        self._vel_min = -self.max_vel
        self._vel_step = (2*self.max_vel)/state_discretization

        cdef int initial_state = self.to_state_id(PendulumState(-pi/4, 0))
        super(InvertedPendulum, self).__init__(state_discretization*state_discretization, action_discretization, 
            {initial_state: 1.0})
        self.observation_space = gym.spaces.Box(low=np.array([0,0,-self.max_vel]), high=np.array([1,1,self.max_vel]), dtype=np.float32)

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()

        # pendulum dynamics
        cdef double g = 10.
        cdef double m = 1.
        cdef double l = 1.
        cdef double dt = 0.05
        cdef double torque = self.action_to_torque(action)
        pstate = self.from_state_id(state)

        newvel = pstate.thetav + (-3*g/(2*l) * sin(pstate.theta + pi) + 3./(m*l**2)*torque) * dt
        newth = pstate.theta + newvel*dt
        newvel = fmax(fmin(newvel, self.max_vel-1e-8), -self.max_vel)
        if newth < -pi:
            newth += 2*pi
        if newth >= pi:
            newth -= 2*pi
        next_state = self.to_state_id(PendulumState(newth, newvel))
        #check_pend = self.from_state_id(next_state)

        self._transition_map.insert(pair[int, double](next_state, 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        cdef double torque = self.action_to_torque(action)
        pstate = self.from_state_id(state)
        # OpenAI gym reward
        cost = pstate.theta ** 2 + 0.1 * (pstate.thetav**2)+ 0.001 * (torque**2)
        max_cost = pi ** 2 + 0.1*self.max_vel**2 + 0.001 * (self.max_torque**2)
        return (-cost + max_cost) / max_cost
    
    cpdef observation(self, int state):
        pstate = self.from_state_id(state)
        return np.array([cos(pstate.theta), sin(pstate.theta), pstate.thetav], dtype=np.float32)

    cdef PendulumState from_state_id(self, int state):
        cdef int th_idx = state % self._state_disc
        cdef int vel_idx = state // self._state_disc
        th = self._state_min + self._state_step * th_idx #self.state_map[th_idx]
        thv = self._vel_min + self._vel_step * vel_idx #self.vel_map[vel_idx]
        return PendulumState(th, thv)

    cdef int to_state_id(self, PendulumState pend_state):
        th = pend_state.theta
        thv = pend_state.thetav
        # round
        cdef int th_round = int(floor((th-self._state_min)/self._state_step))
        cdef int th_vel = int(floor((thv-self._vel_min)/self._vel_step))
        return th_round + self._state_disc * th_vel

    cdef double action_to_torque(self, int action):
        return self.action_map[action]

    cpdef render(self):
        pend_state = self.from_state_id(self.get_state())
        th = pend_state.theta
        thv = pend_state.thetav
        print('(%f, %f) = %d' % (th, thv, self.get_state()))


cdef class MountainCar(TabularEnv):
    """ 
    Dynamics and reward are based on OpenAI gym's implementation of MountainCar-v0
    """

    def __init__(self, int posdisc=64, int veldisc=64, int action_discretization=5):
        self._pos_disc = posdisc
        self._vel_disc = veldisc
        self._action_disc = action_discretization
        self.max_vel = 0.06 # gym 0.07
        self.min_vel = -self.max_vel
        self.max_pos = 0.55 # gym 0.6
        self.min_pos = -1.2 # gym -1.2
        self.goal_pos = 0.5

        self._state_step = (self.max_pos-self.min_pos) / self._pos_disc
        self._vel_step = (self.max_vel-self.min_vel)/self._vel_disc

        cdef int initial_state = self.to_state_id(MountainCarState(-0.5, 0))
        super(MountainCar, self).__init__(self._pos_disc*self._vel_disc, 3, {initial_state: 1.0})
        self.observation_space = gym.spaces.Box(low=np.array([self.min_pos,-self.max_vel]), high=np.array([self.max_pos,self.max_vel]), dtype=np.float32)

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()
        state_vec = self.from_state_id(state)
        position, velocity = state_vec.pos, state_vec.vel
        for _ in range(3):
            velocity += (action-1)*0.001 + cos(3*position)*(-0.0025)
            velocity = fmax(fmin(velocity, self.max_vel-1e-8), self.min_vel)
            position += velocity
            position = fmax(fmin(position, self.max_pos-1e-8), self.min_pos)
            if (position==self.min_pos and velocity<0):
                velocity = 0
        next_state = self.to_state_id(MountainCarState(position, velocity))
        self._transition_map.insert(pair[int, double](next_state, 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        state_vec = self.from_state_id(state)
        if state_vec.pos >= self.goal_pos:
            return 1.0
        return 0.0
    
    cpdef observation(self, int state):
        pstate = self.from_state_id(state)
        return np.array([pstate.pos, pstate.vel], dtype=np.float32)

    cdef MountainCarState from_state_id(self, int state):
        cdef int th_idx = state % self._pos_disc
        cdef int vel_idx = state // self._pos_disc
        th = self.min_pos + self._state_step * th_idx
        thv = self.min_vel + self._vel_step * vel_idx 
        return MountainCarState(th, thv)

    cdef int to_state_id(self, MountainCarState state_vec):
        pos = state_vec.pos
        vel = state_vec.vel
        # round
        cdef int pos_idx = int(floor((pos-self.min_pos)/self._state_step))
        cdef int vel_idx = int(floor((vel-self.min_vel)/self._vel_step))
        return pos_idx + self._pos_disc * vel_idx

    cpdef render(self):
        state_vec = self.from_state_id(self.get_state())
        x1 = state_vec.pos
        x2 = state_vec.vel
        print('(%f, %f) = %d' % (x1, x2, self.get_state()))
