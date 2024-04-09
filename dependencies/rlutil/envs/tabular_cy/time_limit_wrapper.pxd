from debugq.envs cimport env_wrapper

cdef class TimeLimitWrapper(env_wrapper.TabularEnvWrapper):
    cdef int _time_limit
    cdef int _timer
