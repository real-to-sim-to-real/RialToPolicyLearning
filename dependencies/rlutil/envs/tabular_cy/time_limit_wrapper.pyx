# distutils: language=c++

from debugq.envs cimport env_wrapper
from rlutil.envs.tabular_cy cimport tabular_env

cdef class TimeLimitWrapper(env_wrapper.TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv env, int time_limit):
        super(TimeLimitWrapper, self).__init__(env)
        self._timer = 0
        self._time_limit = time_limit
    
    cpdef int reset_state(self):
        self._timer = 0
        return self.wrapped_env.reset_state()

    @property
    def time_limit(self):
        return self._time_limit

    cpdef tabular_env.TimeStep step_state(self, int action):
        ts = self.wrapped_env.step_state(action)
        self._timer += 1
        if self._timer >= self._time_limit:
            ts.done = True
        return ts
