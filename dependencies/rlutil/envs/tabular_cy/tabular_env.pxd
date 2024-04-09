from libcpp.map cimport map


cdef struct TimeStep:
    int state
    double reward
    bint done


cdef class TabularEnv(object):
    cdef public int num_states
    cdef public int num_actions
    cdef public observation_space
    cdef public action_space
    cdef int _state
    cdef public dict initial_state_distribution
    cdef map[int, double] _transition_map
    cpdef transitions(self, int state, int action)
    cdef map[int, double] transitions_cy(self, int state, int action)
    cpdef double reward(self, int state, int action, int next_state)
    cpdef observation(self, int state)
    cpdef step(self, int action)
    cpdef TimeStep step_state(self, int action)
    cpdef reset(self)
    cpdef int reset_state(self)
    cpdef transition_matrix(self)
    cpdef reward_matrix(self)
    cpdef set_state(self, int state)
    cpdef int get_state(self)
    cpdef render(self)

cdef class CliffwalkEnv(TabularEnv):
    cdef double transition_noise

cdef class RandomTabularEnv(TabularEnv):
    cdef double[:,:,:] _transition_matrix
    cdef double[:,:] _reward_matrix


cdef struct PendulumState:
    double theta
    double thetav


cdef class InvertedPendulum(TabularEnv):
    cdef int _state_disc
    cdef int _action_disc
    cdef double max_vel
    cdef double max_torque
    cdef double[:] action_map
    cdef double[:] state_map
    cdef double[:] vel_map
    cdef double _state_min
    cdef double _state_step
    cdef double _vel_min
    cdef double _vel_step
    cdef PendulumState from_state_id(self, int state)
    cdef int to_state_id(self, PendulumState pend_state)
    cdef double action_to_torque(self, int action)

cdef struct MountainCarState:
    double pos
    double vel
    
cdef class MountainCar(TabularEnv):
    cdef int _pos_disc
    cdef int _vel_disc
    cdef int _action_disc
    cdef double max_vel
    cdef double min_vel
    cdef double max_pos
    cdef double min_pos
    cdef double goal_pos
    cdef double _state_step
    cdef double _vel_step
    cdef MountainCarState from_state_id(self, int state)
    cdef int to_state_id(self, MountainCarState pend_state)

