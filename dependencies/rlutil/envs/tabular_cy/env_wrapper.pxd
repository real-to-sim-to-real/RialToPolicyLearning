from rlutil.envs.tabular_cy cimport tabular_env

cdef class TabularEnvWrapper(tabular_env.TabularEnv):
    cdef public tabular_env.TabularEnv wrapped_env

cdef class AbsorbingStateWrapper(TabularEnvWrapper):
    cdef int absorb_state
    cdef double absorb_reward

cdef class StochasticActionWrapper(TabularEnvWrapper):
    cdef double eps
    cdef double eps_new 
    cdef double eps_old 

