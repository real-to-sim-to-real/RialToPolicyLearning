# distutils: language=c++
import numpy as np

from libcpp.map cimport map, pair
from libc.math cimport floor

from rlutil.envs.gridcraft.grid_spec_cy cimport TileType


cdef dict STR_MAP = {
    'O': TileType.EMPTY,
    '#': TileType.WALL,
    'S': TileType.START,
    'R': TileType.REWARD,
    '2': TileType.REWARD2,
    '3': TileType.REWARD3,
    '4': TileType.REWARD4,
    'L': TileType.LAVA
}

RENDER_DICT = {v:k for k, v in STR_MAP.items()}
RENDER_DICT[TileType.EMPTY] = ' '
RENDER_DICT[TileType.START] = ' '


cpdef GridSpec spec_from_string(str s, valmap=STR_MAP):
    if s.endswith('\\'):
        s = s[:-1]
    rows = s.split('\\')
    rowlens = np.array([len(row) for row in rows])
    assert np.all(rowlens == rowlens[0])
    w, h = len(rows[0]), len(rows)

    gs = GridSpec(w, h)
    for i in range(h):
        for j in range(w):
            gs[j,i] = valmap[rows[i][j]]
    return gs


cpdef GridSpec spec_from_sparse_locations(int w, int h, dict tile_to_locs):
    """

    Example usage:
    >> spec_from_sparse_locations(10, 10, {START: [(0,0)], REWARD: [(7,8), (8,8)]})

    """
    gs = GridSpec(w, h)
    for tile_type in tile_to_locs:
        locs = np.array(tile_to_locs[tile_type])
        for i in range(locs.shape[0]):
            gs[tuple(locs[i])] = tile_type
    return gs


cpdef GridSpec local_spec(str map, xpnt):
    """
    >>> local_spec("yOy\\\\Oxy", xpnt=(5,5))
    array([[4, 4],
           [6, 4],
           [6, 5]])
    """
    Y = 0; X=1; O=2
    valmap={
        'y': Y,
        'x': X,
        'O': O
    }
    gs = spec_from_string(map, valmap=valmap)
    ys = gs.find(Y)
    x = gs.find(X)
    result = ys-x + np.array(xpnt)
    return result


cdef class GridSpec(object):
    def __init__(self, int w, int h):
        self.__data_np = np.zeros((w, h), dtype=np.int32)
        self.__data_np[:,:] = TileType.EMPTY
        self.__data = self.__data_np
        self.width = w
        self.height = h

    def __setitem__(self, pair[int, int] key, TileType val):
        self.__data[key.first, key.second] = int(val)

    def __getitem__(self, pair[int, int] key):
        return self.get_value(key)
    
    cpdef TileType get_value(self, pair[int, int] key):
        if self.out_of_bounds(key):
            raise NotImplementedError("Out of bounds: (%d, %d)" % (key.first, key.second))
        return TileType(self.__data[key.first, key.second])

    cpdef bint out_of_bounds(self, pair[int, int] wh):
        """ Return true if x, y is out of bounds """
        w = wh.first
        h = wh.second
        if w<0 or w>=self.width:
            return True
        if h < 0 or h >= self.height:
            return True
        return False

    @property
    def data(self):
        return self.__data_np

    def __len__(self):
        return self.width*self.height

    cpdef pair[int, int] idx_to_xy(self, int idx):
        x = idx % self.width
        y = int(floor(idx/self.width))
        return pair[int, int](x, y)

    cpdef int xy_to_idx(self, pair[int, int] key):
        return key.first + key.second*self.width

    def __hash__(self):
        data = (self.width, self.height) + tuple(self.__data.reshape([-1]).tolist())
        return hash(data)
