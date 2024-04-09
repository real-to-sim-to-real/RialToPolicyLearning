from rlutil.envs.tabular_cy cimport tabular_env
from rlutil.envs.gridcraft cimport grid_spec_cy

cdef class GridEnv(tabular_env.TabularEnv):
    cdef grid_spec_cy.GridSpec gs

cdef class DistanceRewardGridEnv(GridEnv):
    cdef double start_dist
    cdef int rew_x
    cdef int rew_y