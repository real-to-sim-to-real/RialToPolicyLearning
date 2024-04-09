import random
import numpy as np
import scipy as sp
import scipy.stats
import contextlib

def rle(inarray):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        ia = np.array(inarray)                  # force numpy
        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])

def split_list_by_lengths(values, lengths):
    """

    >>> split_list_by_lengths([0,0,0,1,1,1,2,2,2], [2,2,5])
    [[0, 0], [0, 1], [1, 1, 2, 2, 2]]
    """
    assert np.sum(lengths) == len(values)
    idxs = np.cumsum(lengths)
    idxs = np.insert(idxs, 0, 0)
    return [ values[idxs[i]:idxs[i+1] ] for i in range(len(idxs)-1)]


def clip_sing(X, clip_val=1):
    U, E, V = np.linalg.svd(X, full_matrices=False)
    E = np.clip(E, -clip_val, clip_val)
    return U.dot(np.diag(E)).dot(V)


def gauss_log_pdf(params, x):
    mean, log_diag_std = params
    N, d = mean.shape
    cov =  np.square(np.exp(log_diag_std))
    diff = x-mean
    exp_term = -0.5 * np.sum(np.square(diff)/cov, axis=1)
    norm_term = -0.5*d*np.log(2*np.pi)
    var_term = -0.5 * np.sum(np.log(cov), axis=1)
    log_probs = norm_term + var_term + exp_term
    return log_probs #sp.stats.multivariate_normal.logpdf(x, mean=mean, cov=cov)


def categorical_log_pdf(params, x, one_hot=True):
    if not one_hot:
        raise NotImplementedError()
    probs = params[0]
    return np.log(np.max(probs * x, axis=1))


def gd_optimizer(lr, lr_sched=None):
    if lr_sched is None:
        lr_sched = {}

    itr = 0
    def update(x, grad):
        nonlocal itr, lr
        if itr in lr_sched:
            lr *= lr_sched[itr]
        new_x = x - lr * grad
        itr += 1
        return new_x
    return update


def gd_momentum_optimizer(lr, momentum=0.9, lr_sched=None):
    if lr_sched is None:
        lr_sched = {}

    itr = 0
    prev_grad = None
    def update(x, grad):
        nonlocal itr, lr, prev_grad
        if itr in lr_sched:
            lr *= lr_sched[itr]

        if prev_grad is None:
            grad = grad
        else:
            grad = grad + momentum * prev_grad
        new_x = x - lr * grad
        prev_grad = grad
        itr += 1
        return new_x
    return update


def adam_optimizer(lr, beta1=0.9, beta2=0.999, eps=1e-8):
    itr = 0
    pm = None
    pv = None
    def update(x, grad):
        nonlocal itr, lr, pm, pv
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1-beta1)*grad
        pv = beta2 * pv + (1-beta2)*(grad*grad)
        mhat = pm/(1-beta1**(itr+1))
        vhat = pv/(1-beta2**(itr+1))
        update_vec = mhat / (np.sqrt(vhat)+eps)
        new_x = x - lr * update_vec
        itr += 1
        return new_x
    return update


@contextlib.contextmanager
def np_seed(seed):
    """ A context for np random seeds """
    if seed is None:
        yield
    else:
        old_state = np.random.get_state()
        old_py_state = random.getstate()
        np.random.seed(seed)
        random.seed(seed)
        yield
        np.random.set_state(old_state)
        random.setstate(old_py_state)

