# distutils: language=c++
from libcpp.map cimport map, pair

from rlutil.envs.tabular_cy cimport tabular_env

from cython.operator cimport dereference, preincrement

cdef class TabularEnvWrapper(tabular_env.TabularEnv):
    def __init__(self, 
                 tabular_env.TabularEnv wrapped_env):
        self.wrapped_env = wrapped_env
        self.num_states = self.wrapped_env.num_states
        self.num_actions = self.wrapped_env.num_actions
        self.observation_space = self.wrapped_env.observation_space
        self.action_space = self.wrapped_env.action_space
        self.initial_state_distribution = self.wrapped_env.initial_state_distribution

    cdef map[int, double] transitions_cy(self, int state, int action):
        return self.wrapped_env.transitions_cy(state, action)

    cpdef double reward(self, int state, int action, int next_state):
        return self.wrapped_env.reward(state, action, next_state)

    cpdef observation(self, int state):
        return self.wrapped_env.observation(state)

    cpdef tabular_env.TimeStep step_state(self, int action):
        return self.wrapped_env.step_state(action)

    cpdef int reset_state(self):
        return self.wrapped_env.reset_state()

    #cpdef transition_matrix(self):
    #    return self.wrapped_env.transition_matrix()

    #cpdef reward_matrix(self):
    #    return self.wrapped_env.reward_matrix()

    cpdef set_state(self, int state):
        return self.wrapped_env.set_state(state)

    cpdef int get_state(self):
        return self.wrapped_env.get_state()

    cpdef render(self):
        return self.wrapped_env.render()


cdef class AbsorbingStateWrapper(TabularEnvWrapper):
    """A wrapper which moves an agent to an absorbing state after
    receiving reward"""
    def __init__(self, tabular_env.TabularEnv wrapped_env, double absorb_reward=1.0):
        super(AbsorbingStateWrapper, self).__init__(wrapped_env)
        self.num_states += 1
        self.absorb_state = self.num_states - 1
        self.absorb_reward = absorb_reward

    cdef map[int, double] transitions_cy(self, int state, int action):
        if state == self.absorb_state:
            self._transition_map.clear()
            self._transition_map.insert(pair[int, double](self.absorb_state, 1.0))
            return self._transition_map

        old_transitions = self.wrapped_env.transitions_cy(state, action)
        cdef map[int, double] new_transitions
        new_transitions.clear()

        transitions_end = old_transitions.end()
        transitions_it = old_transitions.begin()
        cdef double absorb_prob = 0.0
        while transitions_it != transitions_end:
            next_state = dereference(transitions_it).first
            prob = dereference(transitions_it).second
            if self.reward(state, action, next_state) != 0:
                #TODO(justin): This breaks if the reward depends on next_state
                absorb_prob += prob
                new_transitions.insert(pair[int, double](self.absorb_state, absorb_prob))
            else:
                new_transitions.insert(pair[int, double](next_state, prob))
            preincrement(transitions_it)
        
        # copy new_transitions into old_transitions
        self._transition_map.clear()
        transitions_end = new_transitions.end()
        transitions_it = new_transitions.begin()
        while transitions_it != transitions_end:
            next_state = dereference(transitions_it).first
            prob = dereference(transitions_it).second
            self._transition_map.insert(pair[int, double](next_state, prob))
            preincrement(transitions_it)
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        if state == self.absorb_state:
            return 0.0
        if self.wrapped_env.reward(state, action, next_state) > 0:
            return self.absorb_reward 
        return 0.0

    cpdef tabular_env.TimeStep step_state(self, int action):
        return tabular_env.TabularEnv.step_state(self, action)

    cpdef int reset_state(self):
        return tabular_env.TabularEnv.reset_state(self)

    cpdef set_state(self, int state):
        self.wrapped_env.set_state(state)
        return tabular_env.TabularEnv.set_state(self, state)

    cpdef int get_state(self):
        return tabular_env.TabularEnv.get_state(self)

    #cpdef transition_matrix(self):
    #    return tabular_env.TabularEnv.transition_matrix(self)

    #cpdef reward_matrix(self):
    #    return tabular_env.TabularEnv.reward_matrix(self)


cdef class StochasticActionWrapper(TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv wrapped_env, double eps=0.05):
        super(StochasticActionWrapper, self).__init__(wrapped_env)
        self.eps = eps

        self.eps_new = eps / self.num_actions
        self.eps_old = 1.0 - eps

    cdef map[int, double] transitions_cy(self, int state, int action):
        self._transition_map.clear()

        cdef map[int, double] orig_transitions = self.wrapped_env.transitions_cy(state, action)
        transitions_end = orig_transitions.end()
        transitions_it = orig_transitions.begin()
        while transitions_it != transitions_end:
            next_state = dereference(transitions_it).first
            prob = dereference(transitions_it).second
            self._transition_map.insert(pair[int, double](next_state, prob * self.eps_old))
            preincrement(transitions_it)

        cdef map[int, double] eps_transitions;
        for anew in range(self.num_actions):
            eps_transitions = self.wrapped_env.transitions_cy(state, anew)
            transitions_end = eps_transitions.end()
            transitions_it = eps_transitions.begin()
            while transitions_it != transitions_end:
                next_state = dereference(transitions_it).first
                prob = dereference(transitions_it).second

                it = eps_transitions.find(next_state)
                if it != transitions_end:
                    existing_prob = self._transition_map[next_state]
                else:
                    existing_prob = 0.0

                new_prob = existing_prob + self.eps_new * prob
                self._transition_map[next_state] = new_prob
                preincrement(transitions_it)

        return self._transition_map

