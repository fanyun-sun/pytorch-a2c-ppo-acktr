import numpy as np
import os

def log_saturation(fname, first, relus):
    with open(fname, 'a+') as f:
        if first:
            if os.path.exists(fname): os.remove(fname)
            f.write(','.join(['relu{}'.format(i) for i in range(len(relus))]) + '\n')

        ret = [get_saturation(relu) for relu in relus]
        f.write(','.join(list(map(str, ret))) + '\n')
    return ret


def get_saturation(x):
    """
    input:
        x: 2 dimensional numpy array, [batch, ]
    """
    ln = x.shape[0]
    ret = x[0, :].flatten() <= 0.
    for i in range(1, ln):
        ret = np.logical_and(ret, x[i, :].flatten() <= 0.)

    return ret.sum()/x[0, :].size

def incremental_update(last_values, cur_values):
    beta = .999
    return [(last_value * beta + cur_value * (1-beta)) for last_value, cur_value in zip(last_values, cur_values)]

def adam_rescaling(scale, optimizer):
    for group in optimizer.param_groups:
        for idx in range(len(group['params'])):
            p = group['params'][idx]
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if idx % 2 == 0:
                state['exp_avg']    *= (scale ** (1 / (len(group['params'])/2)))
                state['exp_avg_sq'] *= (scale ** (2 / (len(group['params'])/2)))
            else:
                state['exp_avg']    *= (scale ** ((idx // 2 + 1) / (len(group['params'])/2)))
                state['exp_avg_sq'] *= (scale ** ((idx // 2 + 1) * 2 / (len(group['params'])/2)))

def rmsprop_rescaling(scale, optimizer):
    for group in optimizer.param_groups:
        for idx in range(len(group['params'])):
            p = group['params'][idx]
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if idx % 2 == 0:
                # state['exp_avg']    *= (scale ** (1 / (len(group['params'])/2)))
                state['square_avg'] *= (scale ** (2 / (len(group['params'])/2)))
            else:
                # state['exp_avg']    *= (scale ** ((idx // 2 + 1) / (len(group['params'])/2)))
                state['square_avg'] *= (scale ** ((idx // 2 + 1) * 2 / (len(group['params'])/2)))
