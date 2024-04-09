# distutils: language=c++
import sys

import numpy as np
import gym
import gym.spaces

from rlutil.envs.gridcraft.grid_spec_cy cimport TileType, GridSpec
from rlutil.envs.gridcraft.grid_spec_cy import RENDER_DICT
from rlutil.envs.gridcraft.utils import one_hot_to_flat, flat_to_one_hot
from rlutil.envs.tabular_cy cimport tabular_env

from libcpp.map cimport map, pair
from libc.math cimport abs, fmax

cdef int ACT_NOOP = 0
cdef int ACT_UP = 1
cdef int ACT_DOWN = 2
cdef int ACT_LEFT = 3
cdef int ACT_RIGHT = 4
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

cdef class GridEnv(tabular_env.TabularEnv):
    def __init__(self, 
                 GridSpec gridspec):
        start_xys = np.array(np.where(gridspec.data == TileType.START)).T
        start_idxs = [gridspec.xy_to_idx(pair[int, int](xy[0], xy[1])) for xy in start_xys]
        initial_state_distr = {state: 1.0/len(start_idxs) for state in start_idxs}
        super(GridEnv, self).__init__(len(gridspec), 5, initial_state_distr)
        self.gs = gridspec

    cdef map[int, double] transitions_cy(self, int state, int action):
        cdef int new_x, new_y
        self._transition_map.clear()
        xy = self.gs.idx_to_xy(state)
        tile_type = self.gs.get_value(xy)
        if tile_type == TileType.LAVA or tile_type == TileType.WALL: # Lava gets you stuck
            self._transition_map.insert(pair[int, double](state, 1.0))
        else:
            new_x = xy.first
            new_y = xy.second
            if action == ACT_RIGHT:
                new_x += 1
            elif action == ACT_LEFT:
                new_x -= 1
            elif action == ACT_UP:
                new_y -= 1
            elif action == ACT_DOWN:
                new_y += 1
            new_x = min(max(new_x, 0), self.gs.width-1)
            new_y = min(max(new_y, 0), self.gs.height-1)
            new_xy = pair[int, int](new_x, new_y)
            if self.gs.get_value(new_xy) == TileType.WALL:
                self._transition_map.insert(pair[int, double](state, 1.0))
            else:
                self._transition_map.insert(pair[int, double](self.gs.xy_to_idx(new_xy), 1.0))
        return self._transition_map

    cpdef double reward(self, int state, int action, int next_state):
        cdef TileType tile
        tile = self.gs.get_value(self.gs.idx_to_xy(state))
        if tile == TileType.REWARD:
            return 1.0
        elif tile == TileType.LAVA:
            return -1.0
        return 0.0

    cpdef render(self):
        ostream = sys.stdout
        state = self.get_state()
        ostream.write('-'*(self.gs.width+2)+'\n')
        for h in range(self.gs.height):
            ostream.write('|')
            for w in range(self.gs.width):
                if self.gs.xy_to_idx((w,h)) == state:
                    ostream.write('*')
                else:
                    val = self.gs[w, h]
                    ostream.write(RENDER_DICT[val])
            ostream.write('|\n')
        ostream.write('-' * (self.gs.width + 2)+'\n')


cdef class DistanceRewardGridEnv(GridEnv):
    """ 
    A dense reward gridworld where rewards distances to a goal.
    """
    def __init__(self, 
                 GridSpec gridspec, int reward_x, int reward_y, int start_x, int start_y):
        super(DistanceRewardGridEnv, self).__init__(gridspec)
        self.rew_x = reward_x
        self.rew_y = reward_y
        #self.start_x = reward_x
        #self.start_y = reward_y
        self.start_dist = (abs(start_x - self.rew_x) + abs(start_y - self.rew_y))

    cpdef double reward(self, int state, int action, int next_state):
        cdef pair[int, int] xy_current
        xy_current = self.gs.idx_to_xy(state)

        # distance to goal
        cdef double dist_goal = (abs(self.rew_x - xy_current.first) + abs(self.rew_y - xy_current.second))

        # normalize distance, clip from below to 0.0
        return fmax(0.0, (- dist_goal/self.start_dist) + 1.0)
