from libcpp.map cimport map, pair

cpdef enum TileType:
    EMPTY = 110
    WALL = 111
    START = 112
    REWARD = 113
    OUT_OF_BOUNDS = 114
    REWARD2 = 115
    REWARD3 = 116
    REWARD4 = 117
    LAVA = 118


cdef class GridSpec(object):
    cdef int width, height
    cdef __data_np
    cdef int[:,:] __data

    cpdef bint out_of_bounds(self, pair[int, int] wh)
    cpdef TileType get_value(self, pair[int, int] xy)
    cpdef pair[int, int] idx_to_xy(self, int idx)
    cpdef int xy_to_idx(self, pair[int, int] key)
