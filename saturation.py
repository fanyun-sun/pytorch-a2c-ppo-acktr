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
