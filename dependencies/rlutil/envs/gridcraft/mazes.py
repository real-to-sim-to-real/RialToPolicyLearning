from rlutil.envs.gridcraft.grid_spec import spec_from_string, spec_from_sparse_locations, local_spec, \
    START, REWARD, REWARD2, REWARD3


MAZE_ARENA_64 = spec_from_sparse_locations(64, 64, {START: [(32,32)],
                                                 REWARD: [(0,0), (0,63), (63,0), (63,63)]})

MAZE_ARENA_32 = spec_from_sparse_locations(32, 32, {START: [(16,16)],
                                                    REWARD: [(0,0), (0,31), (31,0), (31,31)]})
MAZE_ARENA_16 = spec_from_sparse_locations(16, 16, {START: [(8,8)],
                                                    REWARD: [(0,0), (0,15), (15,0), (15,15)]})

REW_ARENA_64 = spec_from_sparse_locations(64, 64, {START: [(32,32)],
                                                    REWARD: [(4,4)],
                                                     REWARD2: local_spec(xpnt=(4,4), map="yyy\\"+
                                                                                         "yxy\\"+
                                                                                         "yyy"),
                                                     REWARD3: local_spec(xpnt=(4,4), map="yyyyy\\"+
                                                                                         "yOOOy\\"+
                                                                                         "yOxOy\\"+
                                                                                         "yOOOy\\"+
                                                                                         "yyyyy\\"),
                                                     })

REW_ARENA_128 = spec_from_sparse_locations(128, 128, {START: [(64,64)],
                                                     REWARD: [(10,10)],
                                                     REWARD2: local_spec(xpnt=(10,10), map="yyy\\"+
                                                                                           "yxy\\"+
                                                                                           "yyy"),
                                                     REWARD3: local_spec(xpnt=(10,10), map="yyyyy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yOxOy\\"+
                                                                                           "yOOOy\\"+
                                                                                           "yyyyy\\"),
                                                     })

MAZE1 = spec_from_string("SOO\\"+
                         "OOR\\"
                         )

MAZE2 = spec_from_string("SOOOOO\\"+
                         "OOOOOO\\"+
                         "OOOOOO\\"+
                         "OOOOOR\\"
                         )

MAZE3 = spec_from_string("OOOOOO\\"+
                         "OOOOOO\\"+
                         "OOOSOO\\"+
                         "OOOOOO\\"
                         "OOOOOO\\"
                         "OOOOOO\\"
                         )

MAZE_LAVA = spec_from_string("OOOOOOR\\"+
                             "SOLLLLL\\"+
                             "OOOOOOO\\"+
                             "OOOOOO3\\"
                         )
