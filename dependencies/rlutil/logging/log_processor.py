"""
A collection of utility functions for reading and sorting log files
"""
import os
from os import path
import collections
import json
import csv
import numpy as np
import scipy.stats
import pandas
import itertools


def _find_logs_recursive(root_dir):
    """ Iterate log directories recursively. """
    progress_file = path.join(root_dir, 'progress.csv')
    params_file = path.join(root_dir, 'params.json')

    if path.isfile(params_file) and path.isfile(progress_file):
        yield root_dir
    else:
        for dirname in os.listdir(root_dir):
            dirname = path.join(root_dir, dirname)
            if path.isdir(dirname):
                for log_dir in _find_logs_recursive(dirname):
                    yield log_dir


ExperimentLog = collections.namedtuple('ExperimentLog', ['params', 'progress', 'log_dir'])


def iterate_experiments(root_dir, filter_fn=None):
    """
    Returns an iterator through ExperimentLog objects.

    filter_fn is a function which takes in a params dictionary as an argument and returns
    False if the experiment should be skipped. Using this filter argument will be faster than
    python's filter on this iterator because progress file loading will be skipped.

    Args:
        root_dir: String name of root directory to walk through
        filter_fn: (dict) -> bool filter function

    Returns:
        An iterator through ExperimentLog tuples. Contains two fields
            params: A dictionary of experiment parameters
            progress: A dictionary from keys to numpy arrays of logged values
    """
    for log_dir in _find_logs_recursive(root_dir):
        progress_file = path.join(log_dir, 'progress.csv')
        params_file = path.join(log_dir, 'params.json')

        with open(params_file, 'r') as f:
            params = json.load(f)

        if not filter_fn(params):
            continue

        progress_dict = collections.defaultdict(list)
        with open(progress_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in row:
                    progress_dict[key].append(float(row[key]))
        progress = {key:np.array(progress_dict[key]) for key in progress_dict}
        
        if len(progress) == 0:
            print('WARN: empty log file in %s' % log_dir)
            continue

        yield ExperimentLog(params, progress, log_dir)


def filter_params(exp_logs, params_key, params_value):
    return [log for log in exp_logs if log.params[params_key] == params_value]


def partition_params(all_experiments, split_key):
    """
    Partitions parameters into groups according to split_key

    Args:
        all_experiments: An iterator through ExperimentLog objects
        split_key: A string (or tuple) of keys to group by. 

    Returns:
        A dictionary from values of split_key to lists of ExperimentLog objects
    """
    partition_params = collections.defaultdict(list)
    for exp in all_experiments:
        if isinstance(split_key, (list, tuple)):
            exp_key_val = tuple((exp.params[k] for k in split_key))
        else:
            exp_key_val = exp.params[split_key]
        partition_params[exp_key_val].append(exp)
    return partition_params

def normalize_loss(all_experiments, loss_key=None, env_key=None):
    raise NotImplementedError()

def reduce_last(l, **kwargs):
    return l[-1]

def reduce_first(l, **kwargs):
    return l[0]

def reduce_mean(l, **kwargs):
    return np.mean(l)

def reduce_trimmed_mean(l, **kwargs):
    return scipy.stats.trim_mean(l, 0.1)


def to_data_frame(exps, reduce_fn=reduce_last, ignore_params=('uuid', '__clsname__')):
    """
    Convert experiments to a pandas data frame, reducing along the experiment iterations. 

    reduce_fn reduces along the progress file. The default is reduce_last (use the last value logged)
    """
    val_keys = list(exps[0].progress.keys())
    param_keys = list(exps[0].params.keys())
    param_keys = [key for key in param_keys if key not in ignore_params]

    row_key_to_val = collections.defaultdict(list)
    for exp in exps:
        for val_key in val_keys:
            row_key_to_val[val_key].append(reduce_fn(exp.progress[val_key], key=val_key))
        for param_key in param_keys:
            row_key_to_val[param_key].append(exp.params[param_key])
    frame = pandas.DataFrame(data=row_key_to_val)
    return frame


def timewise_data_frame(exps, time_key='iteration', time_max=None, time_min=0, ignore_params=('uuid', '__clsname__')):
    val_keys = list(exps[0].progress.keys())
    param_keys = list(exps[0].params.keys())
    param_keys = [key for key in param_keys if key not in ignore_params]

    row_key_to_val = collections.defaultdict(list)
    for exp in exps:
        exp_len = len(exp.progress[val_keys[0]])
        if time_max is not None:
            exp_len = min(time_max, exp_len)
            
        for timestep in range(time_min, exp_len):
            for val_key in val_keys:
                row_key_to_val[val_key].append(exp.progress[val_key][timestep])
            for param_key in param_keys:
                row_key_to_val[param_key].append(exp.params[param_key])
            row_key_to_val[time_key].append(timestep)
    frame = pandas.DataFrame(data=row_key_to_val)
    return frame


def aggregate_partitions(partitions, reduce_fn=reduce_last, aggregate_fn=reduce_mean):
    """
    Aggregate partitions into a pandas data frame.
    The column keys will be the log values, and the row keys (stored in the 'split_key' column) will be the split key values.

    reduce_fn reduces along the progress file. The default is reduce_last (use the last value logged)
    aggregate_fn reduces along experiments. The default is reduce_mean (average the value of reduce_fn across experiments)
    """
    col_keys = list(partitions.keys()) 
    rows = list(partitions[col_keys[0]][0].progress.keys())

    row_key_to_val = collections.defaultdict(list)
    for col_key in col_keys:
        exps = partitions[col_key]
        # aggregate exps
        aggs = {row_key: aggregate_fn([reduce_fn(exp.progress[row_key], key=row_key) for exp in exps]) for row_key in rows}
        
        for row_key in aggs:
            row_key_to_val[row_key].append(aggs[row_key])
    
    row_key_to_val['split_key'] = col_keys
    frame = pandas.DataFrame(data=row_key_to_val)
    return frame


def reduce_mean_key(frame, col_key):
    col_vals = list(set(frame[col_key]))
    new_rows = []
    for col_val in col_vals:
        exps = frame.loc[frame[col_key] == col_val]
        new_rows.append(exps.mean())
    new_frame = pandas.DataFrame(data=new_rows)
    new_frame[col_key] = col_vals
    return new_frame


def reduce_mean_keys(frame, col_keys):
    ckey2cvals = [list(set(frame[k])) for k in col_keys]
    all_vals = itertools.product(*ckey2cvals)
    new_rows = []
    for val in all_vals:
        cur_frame = frame
        for i in range(len(col_keys)):
            col_key = col_keys[i]
            col_val = val[i]
            cur_frame = cur_frame.loc[cur_frame[col_key] == col_val]
        cur_frame = cur_frame.mean()
        for i in range(len(col_keys)):
            cur_frame[col_keys[i]] = val[i]
        new_rows.append(cur_frame)
    new_frame = pandas.DataFrame(data=new_rows)
    return new_frame


def rename_partitions(frame, mapping, col_key='split_key'):
    col_vals = frame[col_key]
    col_vals_renamed = [mapping.get(col_val, col_val) for col_val in col_vals]
    frame[col_key] = col_vals_renamed
    return frame

def rename_values(frame, mapping):
    columns_renamed = [mapping.get(col_val, col_val) for col_val in frame.columns]
    frame.columns = columns_renamed
    return frame

def label_scatter_points(x, y, val, ax, global_x_offset=0.02, global_y_offset=0.0, offsets={}):
    """Add labels to points in a scatterplot"""
    a = pandas.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        offset_x = global_x_offset
        offset_y = global_y_offset
        k = str(point['val'])
        if k in offsets:
            offset = offsets[k]
            offset_x += offset[0]
            offset_y += offset[1]
        ax.text(point['x']+offset_x, point['y']+offset_y, k)

if __name__ == "__main__":
    import sys
    dirname = sys.argv[1]
    for log in iterate_experiments(dirname):
        print(log)
        break

