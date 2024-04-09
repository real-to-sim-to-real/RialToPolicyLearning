import os
import random
import json
import time
import contextlib
import uuid
import numpy as np
import collections
import time
import datetime
import dateutil.tz

from rlutil.logging.hyperparameterized import extract_hyperparams
import rlutil.logging.logger as rllablogger

THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
RLUTIL_DIR = os.path.dirname(THIS_FILE_DIR)
DATA_DIR = os.path.join(RLUTIL_DIR, 'data')
LOG_BASE_DIR = os.path.expanduser('~/tmp/rlutil_log')

def generate_exp_name(exp_prefix='exp', exp_id='exp', log_base_dir=LOG_BASE_DIR):
    return '%s/%s/%s' % (log_base_dir, exp_prefix, exp_id)


@contextlib.contextmanager
def setup_logger(algo=None, dirname=None, exp_prefix='exp', log_base_dir=LOG_BASE_DIR):
    reset_logger()
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    exp_uuid = now.strftime('%Y_%m_%d_%H_%M_%S')
    # exp_uuid = str(uuid.uuid4()
    if dirname is None:
        dirname = generate_exp_name(exp_prefix=exp_prefix, exp_id=exp_uuid, log_base_dir=log_base_dir)
    rllablogger.set_snapshot_dir(dirname)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            params['uuid'] = exp_uuid
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))


import traceback 

def save_exception():
    exc_file = os.path.join(rllablogger.get_snapshot_dir(), 'exception.txt')
    with open(exc_file, 'w') as f:
        traceback.print_exc(file=f)
    traceback.print_exc()



MEAN = 'mean'
MAX = 'max'
MIN = 'min'

def record_tabular_stats(key, array, stats=(MEAN, MAX, MIN)):
    if MEAN in stats:
        rllablogger.record_tabular(key+'_mean', np.mean(array))
    if MAX in stats:
        rllablogger.record_tabular(key+'_max', np.max(array))
    if MIN in stats:
        rllablogger.record_tabular(key+'_min', np.min(array))


KEY_TO_VALUES = collections.defaultdict(list)

def record_tabular_moving(key, value, n=100, fill_value=0.0):
    vals = KEY_TO_VALUES[key]
    if len(vals) == 0:
        vals.extend([fill_value]*n)
    vals.append(value)
    vals = vals[-n:]
    KEY_TO_VALUES[key] = vals
    rllablogger.record_tabular(key+'_%d_step_mean' % n, np.mean(vals))

def reset_logger():
    rllablogger.reset()
    KEY_TO_VALUES.clear()


class SubTimer(object):
    def __init__(self):
        self._times = collections.defaultdict(float)
    
    @contextlib.contextmanager
    def subtimer(self, name):
        start = time.time()
        yield
        total = time.time()-start
        self._times[name] += total

@contextlib.contextmanager
def timer(name):
    rllablogger.log('TIMER BEGIN | %s' % name)
    start = time.time()
    subtimer = SubTimer()
    yield subtimer
    total = time.time()-start
    rllablogger.log('TIMER  END  | %s | %fs' % (name, total))
    for val in subtimer._times:
        rllablogger.log('\t SUBTIME | %s | %fs' % (val, subtimer._times[val]))

