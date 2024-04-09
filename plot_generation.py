import numpy as np
import wandb
import plotly.express as px
import plotly.io as pio
import plotly.graph_objs as go
import pandas as pd
import argparse
from scipy import interpolate
import csv
api = wandb.Api()
#from dexhand.utils.plot import extract_run_ids, get_mean_std_bound, interpolate_xy, hex_to_rgb
def extract_run_ids(run_dict):
    method_runs = {}
    for method in run_dict.keys():
        runs = []
        for run_name in run_dict[method]:
            run_history = api.run(run_name).history()
            runs.append(run_history)
        method_runs[method] = runs

    return method_runs

def hex_to_rgb(hex):
    h = hex.lstrip('#')
    color = np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])
    return color

def get_mean_std_bound(new_ys, type_bound="std"):
    if len(new_ys) == 1:
        return new_ys[0], new_ys[0], new_ys[0]
    else:
        mean = np.mean(new_ys, axis=0)

        if type_bound == "std":
            print("std", np.std(new_ys, axis=0))
            upper = mean  + np.std(new_ys, axis=0)
            lower = mean - np.std(new_ys, axis=0)
        elif type_bound == "max":
            upper = np.max(new_ys, axis=0)
            lower = np.min(new_ys, axis=0)
        return mean, upper, lower

def interpolate_xy(old_x, old_y, new_x):
    f = interpolate.interp1d(old_x, old_y)
    y_new = f(new_x)
    return y_new

all_experiments = {
    'drawing':{
        'Ours':['drawing_eval.csv']
    },
    'complex_maze': { 
        'Inverse models': ['locobot-learn/complex_mazegcsl_preferences/3m9swmhw', 'locobot-learn/complex_mazegcsl_preferences/2b3rot23', 'locobot-learn/complex_mazegcsl_preferences/2aszmug5', 'locobot-learn/complex_mazegcsl_preferences/1rygv6yl'],
        'Ours': ['locobot-learn/complex_mazegcsl_preferences/t3ain3iy', 'locobot-learn/complex_mazegcsl_preferences/ql7y35aj', 'locobot-learn/complex_mazegcsl_preferences/19l0ukfl', 'locobot-learn/complex_mazegcsl_preferences/152gn9st'],
        'Oracle': [ 'locobot-learn/complex_mazegcsl_preferences/3ge5tc9z', 'locobot-learn/complex_mazegcsl_preferences/uten8xh0', 'locobot-learn/complex_mazegcsl_preferences/2ol84l4l', 'locobot-learn/complex_mazegcsl_preferences/1md3bl57'],
        'DDL': ['locobot-learn/complex_mazegcsl_preferences/2hofaqx5', 'locobot-learn/complex_mazegcsl_preferences/3lp02y28' ],  # first has to be distance - 5 and will succeed a bit, # the second we have to substract -1
        'Human Preferences':['locobot-learn/complex_mazegcsl_preferences/ace2tq9n', 'locobot-learn/complex_mazegcsl_preferences/1whevytd', 'locobot-learn/complex_mazegcsl_preferences/2fx6yvlm', 'locobot-learn/complex_mazegcsl_preferences/1aftir57'],
        'PPO (sparse)':['locobot-learn/complex_mazegcsl_preferences/3mz2rjde', 'locobot-learn/complex_mazegcsl_preferences/25z4dry2'],
        'PPO (dense)':['locobot-learn/complex_mazegcsl_preferences/20mcgqeh', 'locobot-learn/complex_mazegcsl_preferences/1pdzlzlm'],
        'BC':['locobot-learn/complex_mazegcsl_preferences/2pihz22h','locobot-learn/complex_mazegcsl_preferences/1ecqsetu', 'locobot-learn/complex_mazegcsl_preferences/8qctneul', 'locobot-learn/complex_mazegcsl_preferences/3eso4mth'],
        'BC + Ours':['locobot-learn/complex_mazegcsl_preferences/2pihz22h','locobot-learn/complex_mazegcsl_preferences/1ecqsetu','locobot-learn/complex_mazegcsl_preferences/8qctneul', 'locobot-learn/complex_mazegcsl_preferences/3eso4mth'],
    },
    'human_experiment_pointmass':{ 
        'Ours (human)': ['locobot-learn/pointmass_roomsgcsl_preferences/ff4lhn00'],
        'Human Preferences (human)': ['locobot-learn/pointmass_roomsgcsl_preferences/ugfdv80y'],
        'Ours (synthetic)': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49', 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
    },
    'human_experiment_pusher':{ 
        'Ours (human)': ['locobot-learn/pusher_hardgcsl_preferences/2c5uvl0h'], # rerun
        #'Oracle': ['locobot-learn/pusher_hardgcsl_preferences/33xdekj4','locobot-learn/pusher_hardgcsl_preferences/1rzmvwx7', 'locobot-learn/pusher_hardgcsl_preferences/33xdekj4', 'locobot-learn/pusher_hardgcsl_preferences/55qgljkv'],
        'Ours (synthetic)': ['locobot-learn/pusher_hardgcsl_preferences/rfgqkvnb', 'locobot-learn/pusher_hardgcsl_preferences/1tqpneoc', 'locobot-learn/pusher_hardgcsl_preferences/3j8otoxl', 'locobot-learn/pusher_hardgcsl_preferences/3cfk6e69'],
    },
    'human_experiment_kitchen':{ 
        'Ours (human + 5 demos)': ['locobot-learn/kitchenSeqgcsl_preferences/cco2cpw1'],
        'Ours (crowdsource + 5 demos)': ['locobot-learn/kitchenSeqgcsl_preferences/wszvfglu'], 
        #'Oracle': ['locobot-learn/pusher_hardgcsl_preferences/33xdekj4','locobot-learn/pusher_hardgcsl_preferences/1rzmvwx7', 'locobot-learn/pusher_hardgcsl_preferences/33xdekj4', 'locobot-learn/pusher_hardgcsl_preferences/55qgljkv'],
        'Ours (synthetic + 5 demos)': ['locobot-learn/kitchenSeqgcsl_preferences/ileqzisy', 'locobot-learn/kitchenSeqgcsl_preferences/jhhsktk8','locobot-learn/kitchenSeqgcsl_preferences/udx02pbq','locobot-learn/kitchenSeqgcsl_preferences/gmt39mhk'],
    },
    'adversarial_labels':{ 
        'Oracle':['locobot-learn/pointmass_roomsgcsl_preferences/13i06x5j', 'locobot-learn/pointmass_roomsgcsl_preferences/1gtguy3h', 'locobot-learn/pointmass_roomsgcsl_preferences/1votp9j9', 'locobot-learn/pointmass_roomsgcsl_preferences/2noe2p8j'],
        'Ours': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49', 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
        'Adversarial Labels':['locobot-learn/pointmass_roomsgcsl_preferences/192t8lgl', 'locobot-learn/pointmass_roomsgcsl_preferences/2av2twnl','locobot-learn/pointmass_roomsgcsl_preferences/3mubeuhp'],
        'Human Preferences': ['locobot-learn/pointmass_roomsgcsl_preferences/oxgeth0t']
    },
    'incomplete_goal_selector':{ 
        'Oracle':['locobot-learn/pointmass_roomsgcsl_preferences/13i06x5j', 'locobot-learn/pointmass_roomsgcsl_preferences/1gtguy3h', 'locobot-learn/pointmass_roomsgcsl_preferences/1votp9j9', 'locobot-learn/pointmass_roomsgcsl_preferences/2noe2p8j'],
        'Ours': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49', 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
        'Ours (Incomplete)':['locobot-learn/pointmass_roomsgcsl_preferences/22wok28k', 'locobot-learn/pointmass_roomsgcsl_preferences/24a9yim6', 'locobot-learn/pointmass_roomsgcsl_preferences/3vj8ob4x', 'locobot-learn/pointmass_roomsgcsl_preferences/1gvfvnfx'],
    },
    'pick_and_place':{ 
        'Inverse models': ['locobot-learn/ravens_pick_or_placegcsl_preferences/2z5wg1eb', 'locobot-learn/ravens_pick_or_placegcsl_preferences/2cz1h478', 'locobot-learn/ravens_pick_or_placegcsl_preferences/1f6hv3ln', 'locobot-learn/ravens_pick_or_placegcsl_preferences/1403j7mf'],
        'Oracle': ['locobot-learn/ravens_pick_or_placegcsl_preferences/wmf38v8c', 'locobot-learn/ravens_pick_or_placegcsl_preferences/pdtav62g', 'locobot-learn/ravens_pick_or_placegcsl_preferences/3ml7miyz', 'locobot-learn/ravens_pick_or_placegcsl_preferences/16h3wzvl'],
        'Ours': ['locobot-learn/ravens_pick_or_placegcsl_preferences/kjd9y43s', 'locobot-learn/ravens_pick_or_placegcsl_preferences/2xmt487i', 'locobot-learn/ravens_pick_or_placegcsl_preferences/2cm01vqg', 'locobot-learn/ravens_pick_or_placegcsl_preferences/1pyyabip'],
        'Human Preferences': ['locobot-learn/ravens_pick_or_placegcsl_preferences/2iy1ekvl'] ,
        'DDL':['locobot-learn/ravens_env_pick_or_placegcsl_preferences/2irda94n','locobot-learn/ravens_env_pick_or_placegcsl_preferences/2mq77lqy'],
        'PPO (dense)':['locobot-learn/ravens_env_pick_or_placegcsl_preferences/1b5igcs6', 'locobot-learn/ravens_env_pick_or_placegcsl_preferences/d1hrd74w'],
        'PPO (sparse)':['locobot-learn/ravens_env_pick_or_placegcsl_preferences/12vwmco5', 'locobot-learn/ravens_env_pick_or_placegcsl_preferences/28m2jkne'],
        'BC':['locobot-learn/ravens_pick_placegcsl_preferences/2vrbpakd', 'locobot-learn/ravens_pick_placegcsl_preferences/qt3eu26u', 'locobot-learn/ravens_pick_placegcsl_preferences/3hh3z3z3', 'locobot-learn/ravens_pick_placegcsl_preferences/107mcr5c'],
        'BC + Ours':['locobot-learn/ravens_pick_placegcsl_preferences/2vrbpakd', 'locobot-learn/ravens_pick_placegcsl_preferences/qt3eu26u', 'locobot-learn/ravens_pick_placegcsl_preferences/3hh3z3z3', 'locobot-learn/ravens_pick_placegcsl_preferences/107mcr5c'],
    },
    'bandu_part2': ['locobot-learn/ravens_pick_or_placegcsl_preferences/398w0jmr', 'locobot-learn/ravens_pick_or_placegcsl_preferences/wqnuuub3', 'locobot-learn/ravens_pick_or_placegcsl_preferences/573xow6w','locobot-learn/ravens_pick_or_placegcsl_preferences/2npcwyso'],
    'bandu':{ 
        'Ours': ['locobot-learn/ravens_pick_or_placegcsl_preferences/d1cf2aed', 'locobot-learn/ravens_pick_or_placegcsl_preferences/0a82it2x', 'locobot-learn/ravens_pick_or_placegcsl_preferences/75pc4v93','locobot-learn/ravens_pick_or_placegcsl_preferences/ew8m0wtk'],
        'Inverse models': ['locobot-learn/ravens_pick_or_placegcsl_preferences/yr5ai33l', 'locobot-learn/ravens_pick_or_placegcsl_preferences/wz6ayccf','locobot-learn/ravens_pick_or_placegcsl_preferences/xw88kyy4','locobot-learn/ravens_pick_or_placegcsl_preferences/xy6aquff'],
        'Oracle': ['locobot-learn/ravens_pick_or_placegcsl_preferences/axoaknme','locobot-learn/ravens_pick_or_placegcsl_preferences/8fe5r3vw','locobot-learn/ravens_pick_or_placegcsl_preferences/bbaq48iv','locobot-learn/ravens_pick_or_placegcsl_preferences/o5fjkj7m'],
        'Human Preferences': ['locobot-learn/ravens_pick_or_placegcsl_preferences/2uy65yi9','locobot-learn/ravens_pick_or_placegcsl_preferences/2kdp7g3j','locobot-learn/ravens_pick_or_placegcsl_preferences/66c7gzlh','locobot-learn/ravens_pick_or_placegcsl_preferences/1wc9bmsu'] ,
        'DDL':['locobot-learn/ravens_pick_or_placegcsl_preferences/y1slrbzw', "locobot-learn/ravens_pick_or_placegcsl_preferences/3r3qd2bp", ],
        'PPO (dense)':['locobot-learn/ravens_pick_or_placegcsl_preferences/1u57zp7f','locobot-learn/ravens_pick_or_placegcsl_preferences/1powout8','locobot-learn/ravens_pick_or_placegcsl_preferences/2aaym3uv','locobot-learn/ravens_pick_or_placegcsl_preferences/1hfrdrak'],
        'PPO (sparse)':['locobot-learn/ravens_pick_or_placegcsl_preferences/12mi9vnr','locobot-learn/ravens_pick_or_placegcsl_preferences/36iqrknv','locobot-learn/ravens_pick_or_placegcsl_preferences/guq4rxf0','locobot-learn/ravens_pick_or_placegcsl_preferences/2g3cdm4c'],
        'BC':['locobot-learn/ravens_pick_or_placegcsl_preferences/773nhysm','locobot-learn/ravens_pick_or_placegcsl_preferences/2karmvcg','locobot-learn/ravens_pick_or_placegcsl_preferences/bj63wbex','locobot-learn/ravens_pick_or_placegcsl_preferences/mg2rj75c'],
        'BC + Ours':['locobot-learn/ravens_pick_or_placegcsl_preferences/773nhysm','locobot-learn/ravens_pick_or_placegcsl_preferences/2karmvcg','locobot-learn/ravens_pick_or_placegcsl_preferences/bj63wbex','locobot-learn/ravens_pick_or_placegcsl_preferences/mg2rj75c'],
    },
    'kitchen':{ 
        'Inverse models': ['locobot-learn/kitchenSeqgcsl_preferences/0oja1kgk','locobot-learn/kitchenSeqgcsl_preferences/hrqgvp9k','locobot-learn/kitchenSeqgcsl_preferences/wvix2h1o','locobot-learn/kitchenSeqgcsl_preferences/j5b9kjmn'],
        'Oracle': ['locobot-learn/kitchenSeqgcsl_preferences/9g4xanhf','locobot-learn/kitchenSeqgcsl_preferences/f7lc22yj','locobot-learn/kitchenSeqgcsl_preferences/wysnrw02','locobot-learn/kitchenSeqgcsl_preferences/01l23j5u'],
        'Ours': ['locobot-learn/kitchenSeqgcsl_preferences/cbmrcbxd','locobot-learn/kitchenSeqgcsl_preferences/54i2szv8','locobot-learn/kitchenSeqgcsl_preferences/qkmp1j34','locobot-learn/kitchenSeqgcsl_preferences/s64v24xb'],
        'DDL': ['locobot-learn/kitchenSeqgcsl_preferences/1kr62web', 'locobot-learn/kitchenSeqgcsl_preferences/28uem0sc'],
        'PPO (dense)':['locobot-learn/kitchenSeqgcsl_preferences/lo7yjrio','locobot-learn/kitchenSeqgcsl_preferences/ta22u7ts','locobot-learn/kitchenSeqgcsl_preferences/y7dgf7kc','locobot-learn/kitchenSeqgcsl_preferences/10fp2825'],
        'PPO (sparse)':['locobot-learn/kitchenSeqgcsl_preferences/3n71xlqv','locobot-learn/kitchenSeqgcsl_preferences/18j65tu4','locobot-learn/kitchenSeqgcsl_preferences/3onew6ai','locobot-learn/kitchenSeqgcsl_preferences/kh7g5hxf'],
        'Human Preferences': ['locobot-learn/kitchenSeqgcsl_preferences/1hm0k2qg','locobot-learn/kitchenSeqgcsl_preferences/1hvxmexj','locobot-learn/kitchenSeqgcsl_preferences/3flif3st','locobot-learn/kitchenSeqgcsl_preferences/2dlq4umx'], 
        'BC':['locobot-learn/kitchenSeqgcsl_preferences/ileqzisy','locobot-learn/kitchenSeqgcsl_preferences/jhhsktk8','locobot-learn/kitchenSeqgcsl_preferences/udx02pbq','locobot-learn/kitchenSeqgcsl_preferences/gmt39mhk'],
        'BC + Ours':['locobot-learn/kitchenSeqgcsl_preferences/ileqzisy','locobot-learn/kitchenSeqgcsl_preferences/jhhsktk8','locobot-learn/kitchenSeqgcsl_preferences/udx02pbq','locobot-learn/kitchenSeqgcsl_preferences/gmt39mhk'],
    },
    'pusher_walls':{ 
        'Oracle': ['locobot-learn/pusher_hardgcsl_preferences/33xdekj4','locobot-learn/pusher_hardgcsl_preferences/1rzmvwx7', 'locobot-learn/pusher_hardgcsl_preferences/33xdekj4', 'locobot-learn/pusher_hardgcsl_preferences/55qgljkv'],
        'Inverse models': ['locobot-learn/pusher_hardgcsl_preferences/23ojli2z', 'locobot-learn/pusher_hardgcsl_preferences/1pydiqsr', 'locobot-learn/pusher_hardgcsl_preferences/2w47tg7c', 'locobot-learn/pusher_hardgcsl_preferences/32ksqe2q'],
        'Ours': ['locobot-learn/pusher_hardgcsl_preferences/rfgqkvnb', 'locobot-learn/pusher_hardgcsl_preferences/1tqpneoc', 'locobot-learn/pusher_hardgcsl_preferences/3j8otoxl', 'locobot-learn/pusher_hardgcsl_preferences/3cfk6e69'],
        'DDL':['locobot-learn/pusher_hardgcsl_preferences/2hvf8n26', 'locobot-learn/pusher_hardgcsl_preferences/2vgo48q0', 'locobot-learn/pusher_hardgcsl_preferences/5sm4dxbk', 'locobot-learn/pusher_hardgcsl_preferences/3qljdbvw'],
        'PPO (dense)': ['locobot-learn/pusher_hardgcsl_preferences/1rudeyr3','locobot-learn/pusher_hardgcsl_preferences/3s2csd9t'],
        'PPO (sparse)':['locobot-learn/pusher_hardgcsl_preferences/1j7kfuze', 'locobot-learn/pusher_hardgcsl_preferences/jyqg84p6'],
        'Human Preferences':['locobot-learn/pusher_hardgcsl_preferences/wzoym8xu', 'locobot-learn/pusher_hardgcsl_preferences/wv1kft7m',],
        'BC':['locobot-learn/pusher_hardgcsl_preferences/p5vejh7p','locobot-learn/pusher_hardgcsl_preferences/28kg1171','locobot-learn/pusher_hardgcsl_preferences/7pho1myk','locobot-learn/pusher_hardgcsl_preferences/opj12gx1'],
        #'BC':['locobot-learn/pusher_hardgcsl_preferences/314mfcgf','locobot-learn/pusher_hardgcsl_preferences/1ayweq4j','locobot-learn/pusher_hardgcsl_preferences/340z7bi5','locobot-learn/pusher_hardgcsl_preferences/2dz0kkx6'],
        'BC + Ours':['locobot-learn/pusher_hardgcsl_preferences/p5vejh7p','locobot-learn/pusher_hardgcsl_preferences/28kg1171','locobot-learn/pusher_hardgcsl_preferences/7pho1myk','locobot-learn/pusher_hardgcsl_preferences/opj12gx1'],
        #'BC + Ours':['locobot-learn/pusher_hardgcsl_preferences/314mfcgf','locobot-learn/pusher_hardgcsl_preferences/1ayweq4j','locobot-learn/pusher_hardgcsl_preferences/340z7bi5','locobot-learn/pusher_hardgcsl_preferences/2dz0kkx6'],
    },
    'pointmass_rooms':{ 
        'BC':['locobot-learn/pointmass_roomsgcsl_preferences/y3pcrcfc','locobot-learn/pointmass_roomsgcsl_preferences/3m4my8sp','locobot-learn/pointmass_roomsgcsl_preferences/2yz5vu1l','locobot-learn/pointmass_roomsgcsl_preferences/2m83rfwp'],
        'BC + Ours':['locobot-learn/pointmass_roomsgcsl_preferences/y3pcrcfc','locobot-learn/pointmass_roomsgcsl_preferences/3m4my8sp','locobot-learn/pointmass_roomsgcsl_preferences/2yz5vu1l','locobot-learn/pointmass_roomsgcsl_preferences/2m83rfwp'],
        'Inverse models': ['locobot-learn/pointmass_roomsgcsl_preferences/qppe50f8', 'locobot-learn/pointmass_roomsgcsl_preferences/grp4fz5x','locobot-learn/pointmass_roomsgcsl_preferences/1bzq2ql5', 'locobot-learn/pointmass_roomsgcsl_preferences/21te32ys'],
        'Oracle': ['locobot-learn/pointmass_roomsgcsl_preferences/13i06x5j', 'locobot-learn/pointmass_roomsgcsl_preferences/1gtguy3h', 'locobot-learn/pointmass_roomsgcsl_preferences/1votp9j9', 'locobot-learn/pointmass_roomsgcsl_preferences/2noe2p8j'],
        'Ours': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49', 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
        'DDL': ['locobot-learn/pointmass_roomsgcsl_preferences/2ilprc3u', 'locobot-learn/pointmass_roomsgcsl_preferences/2ilprc3u'],
        'Human Preferences':['locobot-learn/pointmass_roomsgcsl_preferences/hn5zvi9a', 'locobot-learn/pointmass_roomsgcsl_preferences/3p9qgl7z', 'locobot-learn/pointmass_roomsgcsl_preferences/3gf5fykq'],
        'PPO (dense)': ['locobot-learn/pointmass_roomsgcsl_preferences/2v6od2nu', 'locobot-learn/pointmass_roomsgcsl_preferences/2vjygxhw', 'locobot-learn/pointmass_roomsgcsl_preferences/11yl3qwp'],
        'PPO (sparse)': ['locobot-learn/pointmass_roomsgcsl_preferences/azvlfosm', 'locobot-learn/pointmass_roomsgcsl_preferences/33uelilf'],
        'LEXA-like':['locobot-learn/pointmass_roomsgcsl_preferences/nexy6rfs', 'locobot-learn/pointmass_roomsgcsl_preferences/3omlhojk', 'locobot-learn/pointmass_roomsgcsl_preferences/tm5qy6ow', 'locobot-learn/pointmass_roomsgcsl_preferences/2419021b'],
        },

    # 'pointmass_rooms':{ 
    #     'BC':['locobot-learn/pointmass_roomsgcsl_preferences/y3pcrcfc'],#'locobot-learn/pointmass_roomsgcsl_preferences/3m4my8sp','locobot-learn/pointmass_roomsgcsl_preferences/2yz5vu1l','locobot-learn/pointmass_roomsgcsl_preferences/2m83rfwp'],
    #     'BC + Ours':['locobot-learn/pointmass_roomsgcsl_preferences/y3pcrcfc'],#'locobot-learn/pointmass_roomsgcsl_preferences/3m4my8sp','locobot-learn/pointmass_roomsgcsl_preferences/2yz5vu1l','locobot-learn/pointmass_roomsgcsl_preferences/2m83rfwp'],
    #     'Inverse models': ['locobot-learn/pointmass_roomsgcsl_preferences/qppe50f8'],# 'locobot-learn/pointmass_roomsgcsl_preferences/grp4fz5x','locobot-learn/pointmass_roomsgcsl_preferences/1bzq2ql5', 'locobot-learn/pointmass_roomsgcsl_preferences/21te32ys'],
    #     'Oracle': ['locobot-learn/pointmass_roomsgcsl_preferences/13i06x5j'],# 'locobot-learn/pointmass_roomsgcsl_preferences/1gtguy3h', 'locobot-learn/pointmass_roomsgcsl_preferences/1votp9j9', 'locobot-learn/pointmass_roomsgcsl_preferences/2noe2p8j'],
    #     'Ours': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49'],# 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
    #     'DDL': ['locobot-learn/pointmass_roomsgcsl_preferences/2ilprc3u'],# 'locobot-learn/pointmass_roomsgcsl_preferences/2ilprc3u'],
    #     'Human Preferences':['locobot-learn/pointmass_roomsgcsl_preferences/hn5zvi9a'],# 'locobot-learn/pointmass_roomsgcsl_preferences/3p9qgl7z', 'locobot-learn/pointmass_roomsgcsl_preferences/3gf5fykq'],
    #     'PPO (dense)': ['locobot-learn/pointmass_roomsgcsl_preferences/2v6od2nu'],# 'locobot-learn/pointmass_roomsgcsl_preferences/2vjygxhw', 'locobot-learn/pointmass_roomsgcsl_preferences/11yl3qwp'],
    #     'PPO (sparse)': ['locobot-learn/pointmass_roomsgcsl_preferences/azvlfosm'],# 'locobot-learn/pointmass_roomsgcsl_preferences/33uelilf'],
    #     'LEXA-like':['locobot-learn/pointmass_roomsgcsl_preferences/nexy6rfs'],# 'locobot-learn/pointmass_roomsgcsl_preferences/3omlhojk', 'locobot-learn/pointmass_roomsgcsl_preferences/tm5qy6ow', 'locobot-learn/pointmass_roomsgcsl_preferences/2419021b'],
    #     },
    'freq_labels': { # TODO add xlim for freq 15 (because of drop)
        '1': ['locobot-learn/pointmass_roomsgcsl_preferences/1ky96rdv', 'locobot-learn/pointmass_roomsgcsl_preferences/2h3e20u3', 'locobot-learn/pointmass_roomsgcsl_preferences/36e9llgu', 'locobot-learn/pointmass_roomsgcsl_preferences/l6vwgm33'],
        '15': ['locobot-learn/pointmass_roomsgcsl_preferences/3f85h2po', 'locobot-learn/pointmass_roomsgcsl_preferences/17sz6kcq', 'locobot-learn/pointmass_roomsgcsl_preferences/288yv7s1', 'locobot-learn/pointmass_roomsgcsl_preferences/3j0mcd6g'],
        '100': ['locobot-learn/pointmass_roomsgcsl_preferences/2tsq2du0', 'locobot-learn/pointmass_roomsgcsl_preferences/39wyxe0m', 'locobot-learn/pointmass_roomsgcsl_preferences/38a29e1m', 'locobot-learn/pointmass_roomsgcsl_preferences/2ply3a5k'],
        '500': ['locobot-learn/pointmass_roomsgcsl_preferences/11pm1jky', 'locobot-learn/pointmass_roomsgcsl_preferences/ug533i57', 'locobot-learn/pointmass_roomsgcsl_preferences/3r57px8k', 'locobot-learn/pointmass_roomsgcsl_preferences/392j5lx7']
    },
    'oracle_vs_ours' : {
        '1 (ours)': ['locobot-learn/pointmass_roomsgcsl_preferences/1ky96rdv', 'locobot-learn/pointmass_roomsgcsl_preferences/2h3e20u3', 'locobot-learn/pointmass_roomsgcsl_preferences/36e9llgu', 'locobot-learn/pointmass_roomsgcsl_preferences/l6vwgm33'],
        '15 (ours)': ['locobot-learn/pointmass_roomsgcsl_preferences/3f85h2po', 'locobot-learn/pointmass_roomsgcsl_preferences/17sz6kcq', 'locobot-learn/pointmass_roomsgcsl_preferences/288yv7s1', 'locobot-learn/pointmass_roomsgcsl_preferences/3j0mcd6g'],
        '100 (ours)': ['locobot-learn/pointmass_roomsgcsl_preferences/2tsq2du0', 'locobot-learn/pointmass_roomsgcsl_preferences/39wyxe0m', 'locobot-learn/pointmass_roomsgcsl_preferences/38a29e1m', 'locobot-learn/pointmass_roomsgcsl_preferences/2ply3a5k'],
        '500 (ours)': ['locobot-learn/pointmass_roomsgcsl_preferences/11pm1jky', 'locobot-learn/pointmass_roomsgcsl_preferences/ug533i57', 'locobot-learn/pointmass_roomsgcsl_preferences/3r57px8k', 'locobot-learn/pointmass_roomsgcsl_preferences/392j5lx7'],
        '1 (oracle)': ['locobot-learn/pointmass_roomsgcsl_preferences/1nfhd4x8', 'locobot-learn/pointmass_roomsgcsl_preferences/21iw77ab', 'locobot-learn/pointmass_roomsgcsl_preferences/1e9ao2dm', 'locobot-learn/pointmass_roomsgcsl_preferences/38fvbk7h'],
        '15 (oracle)': ['locobot-learn/pointmass_roomsgcsl_preferences/31w3dwy5', 'locobot-learn/pointmass_roomsgcsl_preferences/6qrr2d06', 'locobot-learn/pointmass_roomsgcsl_preferences/266aagpj', 'locobot-learn/pointmass_roomsgcsl_preferences/2r0qykzf'],
        '100 (oracle)': ['locobot-learn/pointmass_roomsgcsl_preferences/38sayn6v', 'locobot-learn/pointmass_roomsgcsl_preferences/3o737ja5', 'locobot-learn/pointmass_roomsgcsl_preferences/11zoriu6', 'locobot-learn/pointmass_roomsgcsl_preferences/3jv6k4h4'],
        '500 (oracle)': ['locobot-learn/pointmass_roomsgcsl_preferences/2fkm96m0', 'locobot-learn/pointmass_roomsgcsl_preferences/3umfpkwo', 'locobot-learn/pointmass_roomsgcsl_preferences/388roqit', 'locobot-learn/pointmass_roomsgcsl_preferences/3jvhsogn'],
    },
    'num_labels':{
        '1':['locobot-learn/pointmass_roomsgcsl_preferences/39riw8xa', 'locobot-learn/pointmass_roomsgcsl_preferences/2lsgu90a', 'locobot-learn/pointmass_roomsgcsl_preferences/22l19k95', 'locobot-learn/pointmass_roomsgcsl_preferences/1qobpbpd'],
        #'3':['locobot-learn/pointmass_roomsgcsl_preferences/v38qz3zf', 'locobot-learn/pointmass_roomsgcsl_preferences/mzlj5q38', 'locobot-learn/pointmass_roomsgcsl_preferences/2i3qu7yf', 'locobot-learn/pointmass_roomsgcsl_preferences/1urljug4'],
        '5':['locobot-learn/pointmass_roomsgcsl_preferences/1junackm', 'locobot-learn/pointmass_roomsgcsl_preferences/20ed3lar', 'locobot-learn/pointmass_roomsgcsl_preferences/31g3albx', 'locobot-learn/pointmass_roomsgcsl_preferences/2wzniwy5'],
        '20':['locobot-learn/pointmass_roomsgcsl_preferences/1zz2yoy4', 'locobot-learn/pointmass_roomsgcsl_preferences/1b1gppta', 'locobot-learn/pointmass_roomsgcsl_preferences/2rhsd97d', 'locobot-learn/pointmass_roomsgcsl_preferences/3sk4crk1'],
        '100':['locobot-learn/pointmass_roomsgcsl_preferences/q4rgp0hj', 'locobot-learn/pointmass_roomsgcsl_preferences/372f50x1', 'locobot-learn/pointmass_roomsgcsl_preferences/36cq45ro', 'locobot-learn/pointmass_roomsgcsl_preferences/2j1te44k']
    },
    'noise':{
        '0.05 (ours)':['locobot-learn/pointmass_roomsgcsl_preferences/1lgpgz3a','locobot-learn/pointmass_roomsgcsl_preferences/2z90c7al', 'locobot-learn/pointmass_roomsgcsl_preferences/5ym9n60h', 'locobot-learn/pointmass_roomsgcsl_preferences/2gqq6jy5'],
        '0.1 (ours)':['locobot-learn/pointmass_roomsgcsl_preferences/2pabnt30', 'locobot-learn/pointmass_roomsgcsl_preferences/3978utuq', 'locobot-learn/pointmass_roomsgcsl_preferences/37n3n4j0', 'locobot-learn/pointmass_roomsgcsl_preferences/1qjhjzo3'],
        '0.3 (ours)':['locobot-learn/pointmass_roomsgcsl_preferences/3ioe5coo','locobot-learn/pointmass_roomsgcsl_preferences/562bu822', 'locobot-learn/pointmass_roomsgcsl_preferences/220eu1mf', 'locobot-learn/pointmass_roomsgcsl_preferences/1vy9n9bd'],
        #'0.5 (ours)':['locobot-learn/pointmass_roomsgcsl_preferences/12s0efeb','locobot-learn/pointmass_roomsgcsl_preferences/2jj2kvwi', 'locobot-learn/pointmass_roomsgcsl_preferences/1pcb9skm'],
        '1 (ours)': ['locobot-learn/pointmass_roomsgcsl_preferences/3ml8nzop', 'locobot-learn/pointmass_roomsgcsl_preferences/3jr5th97', 'locobot-learn/pointmass_roomsgcsl_preferences/18nfj4g5', 'locobot-learn/pointmass_roomsgcsl_preferences/13q7bzup'],
        '0.05 (oracle)':['locobot-learn/pointmass_roomsgcsl_preferences/355o1lzi',' locobot-learn/pointmass_roomsgcsl_preferences/jlvvys10', 'locobot-learn/pointmass_roomsgcsl_preferences/16x4tkis', 'locobot-learn/pointmass_roomsgcsl_preferences/3nxw5aj9'],
        '0.1 (oracle)':['locobot-learn/pointmass_roomsgcsl_preferences/5gv48bh1', 'locobot-learn/pointmass_roomsgcsl_preferences/h3ei9hok', 'locobot-learn/pointmass_roomsgcsl_preferences/7mlq75zg', 'locobot-learn/pointmass_roomsgcsl_preferences/bvv6s92j'],
        '0.3 (oracle)':['locobot-learn/pointmass_roomsgcsl_preferences/ytez8jf7', 'locobot-learn/pointmass_roomsgcsl_preferences/wleb3fpr', 'locobot-learn/pointmass_roomsgcsl_preferences/19740s3x', 'locobot-learn/pointmass_roomsgcsl_preferences/c6wi3m6q'],
        #'0.5 (oracle)':['locobot-learn/pointmass_roomsgcsl_preferences/i7ut1h7y'],
        '1 (oracle)': ['locobot-learn/pointmass_roomsgcsl_preferences/1pt2ndtt', 'locobot-learn/pointmass_roomsgcsl_preferences/4v6uzwgk', 'locobot-learn/pointmass_roomsgcsl_preferences/1k665v7n', 'locobot-learn/pointmass_roomsgcsl_preferences/10ib3mns'],
    },
    'incomplete_goal_selector2':{ 
        'Oracle':['locobot-learn/pointmass_roomsgcsl_preferences/13i06x5j', 'locobot-learn/pointmass_roomsgcsl_preferences/1gtguy3h', 'locobot-learn/pointmass_roomsgcsl_preferences/1votp9j9', 'locobot-learn/pointmass_roomsgcsl_preferences/2noe2p8j'],
        'no stopping': ['locobot-learn/pointmass_roomsgcsl_preferences/345hvc49', 'locobot-learn/pointmass_roomsgcsl_preferences/2y72w21i', 'locobot-learn/pointmass_roomsgcsl_preferences/129t3bny', 'locobot-learn/pointmass_roomsgcsl_preferences/33i8coln'],
        'final room':['locobot-learn/pointmass_roomsgcsl_preferences/fumfnunt', 'locobot-learn/pointmass_roomsgcsl_preferences/3r3vl2hk'],
        'third room':['locobot-learn/pointmass_roomsgcsl_preferences/3u8tbjt3', 'locobot-learn/pointmass_roomsgcsl_preferences/2m8a6irs'],
        'second room':['locobot-learn/pointmass_roomsgcsl_preferences/1ttghzd7','locobot-learn/pointmass_roomsgcsl_preferences/1myp08ya'],
        'first room':['locobot-learn/pointmass_roomsgcsl_preferences/1fi1ijcw','locobot-learn/pointmass_roomsgcsl_preferences/33r9057z'],
    },
}

titles = {
    'complex_maze': "Maze",
    'human_experiment_pointmass': "Four Rooms (Human Experiment)",
    'pick_and_place':"Block Stacking",
    'bandu':"Bandu",
    'kitchen': "Kitchen",
    'pusher_walls':"Pusher with walls",
    'pusher_empty': "Pusher",
    'empty_room': "Empty Room",
    'human_experiment_pusher':'Pusher (Human Experiment)',
    'human_experiment_kitchen':'Kitchen (Human Experiment)',
    'pointmass_rooms':'Four Rooms',
    'incomplete_goal_selector':'Analysis Learning a Policy from an Incomplete Goal Selector',
    'incomplete_goal_selector2':'Analysis Learning a Policy from an Incomplete Goal Selector',
    'adversarial_labels':'Analysis with Adversarial Labels',
    'freq_labels': "Ablations on frequency of annotations",
    'num_labels': "Ablations on the number of queries per batch",
    'noise': "Ablations on the injected noise on labels",
    'drawing': "Drawing in the real world"
}
# TODO: fix pusher BC
if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--time', '-t', action='store_true')
    parser.add_argument('--model', '-m', type=str, default='ours')
    parser.add_argument('--xlim', '-x', type=float, default='-1')
    parser.add_argument('--xaxis', type=str, default='timesteps')
    parser.add_argument('--ylim', '-y', type=int, default='1')
    parser.add_argument('--ystart', type=float, default='-0.02')

    parser.add_argument('--n_points', type=int, default='40')
    parser.add_argument('--N', '-N', type=int, default='10')
    parser.add_argument('--experiment', type=str, default='pick_and_place')
    parser.add_argument("--show_legend",action="store_true", default=False)
    args = parser.parse_args()
    # runs_dict = all_experiments[args.experiment]
    # runs_dict = runs_dict
    runs_dict =   { 
        'RL-state': ['locobot-learn/drawerbiggerhandle.usdppo-finetune/l3zbpqta'],
        'RL-vision':['locobot-learn/drawerbiggerhandle.usdppo-finetune/i9vlguk6'],
    }
    #run_ids = extract_run_ids(runs_dict)

    runs = list(runs_dict.keys())

    xlim = 35
    ylim = 1

    keys = ["success"]
   

    api = wandb.Api()
    df_pool = dict()
    scatter_plots = []
    colors = {
        'RL-vision': '#7EC8E3',
        'RL-state': '#7921B1',

        'Human': '#7921B1',
        'Ours (human)': '#7921B1',
        'Ours (human + 5 demos)': '#7921B1',
        'Ours (crowdsource + 5 demos)':'#C55FFC',

        'Oracle': '#FFF017',

        'Ours': '#EE3377',
        'Ours (synthetic)': '#EE3377',
        'Ours (synthetic + 5 demos)': '#EE3377',
        'Ours (5 demos)': '#EE3377',
        
        'Human Preferences':'#808080',
        'Human Preferences (human)':'#808080',
        
        'DDL':'#B4FEE7',

        'PPO (dense)':'#0000FF',

        'Ours (Incomplete)':'#990000',
        
        'Adversarial Labels':'#990000',

        'PPO (sparse)':'#000C66',

        'LEXA-like':'#C197D2',
        'BC':'#234F1E',
        'BC + Ours':'#90EE90',
        

        # Freq
        '1':'#669866',
        '15':'#3D9B83',
        '100':'#0D98BA',
        '500':'#8487C3',

        # Noise
        '0.05 (ours)':'#669866',
        '0.1 (ours)':'#3D9B83',
        '0.3 (ours)':'#0D98BA',
        '0.5 (ours)':'#8487C3',
        '1 (ours)':'#8487C3',
        '0.05 (oracle)':'#F56EB3',
        '0.1 (oracle)':'#CB1C8D',
        '0.3 (oracle)':'#7F167F',
        '0.5 (oracle)':'#460C68',
        '1 (oracle)':'#460C68',

        # Num labels
        '1 num_labels':'#669866',
        '3 num_labels':'#EE3377',
        '5 num_labels':'#3D9B83',
        '20 num_labels':'#0D98BA',
        '100 num_labels':'#8487C3',

        # Stopping
        'final room':'#669866',
        'third room':'#3D9B83',
        'second room':'#0D98BA',
        'first room':'#8487C3',
        'no stopping':'#EE3377',
    }

    # TODO: make PPO dense dotted
    labels_max = []
    for idx, (label, run_ids) in enumerate(runs_dict.items()):
        ys = []
        xs = []
        run_id_idx = 0
        for run_id in run_ids:
            print(run_id)
            run = api.run(run_id)
            keys_to_pass = keys
            N_to_pass = 200

                
           
            print(keys_to_pass)
            history = run.scan_history(keys=keys_to_pass)
            y_val = []
            x_val = []
            for x in history:    
                print(x.keys())
                y_val.append(x[keys_to_pass[0]])
                print(x[keys_to_pass[0]])
            
            y_val = np.array(y_val)
            
            z = np.ones(len(y_val))
            
            if idx == 1:
                x_max = 35.423
            else:
                x_max = 23.124
            y_val = np.array(y_val)

            x_val = np.linspace(0, x_max, len(y_val))            
           
            ys.append(np.array(y_val))
            xs.append(np.array(x_val))
            run_id_idx += 1

            

        # if args.time:
        # a tight bound
        
        xs_min = max([min(x) for x in xs])
        xs_max = min([max(x) for x in xs])
        labels_max.append(xs_max)
        print(f'{label}: x min:{xs_min}   max:{xs_max}')
        
        bound = 50
        newx = np.linspace(xs_min, xs_max, bound)

        new_ys = []
        for i in range(len(xs)):
            new_ys.append(interpolate_xy(xs[i], ys[i], newx))
        
        
        dmean, dlower, dupper = get_mean_std_bound(new_ys)
        df_pool[label] = dict(x=newx, mean=dmean, lower=dlower, upper=dupper)
        label_color = label
        color = colors[label_color]
        showlegend = args.show_legend
        name = label
        scatter_plots.append(
            go.Scatter(
                x=newx,
                y=dmean,
                line=dict(color=color,
                          dash= "dash" if 'BC' == label else 'solid'),
                # line=dict(color=f'rgb({color[0]}, {color[1]}, {color[2]})',
                #           dash='solid' if 'split' not in label else 'dot'),
                mode='lines',
                showlegend=showlegend,
                name=name
            )
        )

        scatter_plots.append(
            go.Scatter(
                x=np.concatenate((newx, newx[::-1])),  # x, then x reversed
                y=np.concatenate((dupper, dlower[::-1])),  # upper, then lower reversed
                fill='toself',
                fillcolor=f'rgba{(*hex_to_rgb(color), 0.15)}', #f'rgba({color[0]}, {color[1]}, {color[2]}, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False
            )
        )

    if xlim < 0:
        xlim = max(labels_max)
    fig = go.Figure(scatter_plots)

    yaxis_title = 'Success Ratio'
    xaxis_title = "Number of steps"

    fig.update_layout(
        width=1600,
        height=900,
        title={
            'text': "RLvision",
            'x': 0.5},
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        font=dict(
            family="Arial",
            size= 45,
        ),
        legend_title=None,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="right",
            x=1.15,
            font=dict(size=45)
        )
    )
    fig.update_xaxes(showline=True, linewidth=8, linecolor='black', gridcolor='#c2c2c2',
                     title_standoff=10,
                     automargin=True,
                     range=[0, xlim]
                     )
    fig.update_yaxes(showline=True, linewidth=8, linecolor='black', range=[-0.02, ylim+0.02],
                     title_standoff=10,
                     automargin=True,
                     gridcolor='#c2c2c2')
    
    fig.update_traces(line=dict(width=8))
    save_dir = "figures/"#pathlib_file('figures')
    dest_file = f'rlvision.pdf'
    fig.write_image(dest_file)
    fig.show()
    # embed()