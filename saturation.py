import numpy as np
import os
def log_saturation(fname, first, relu1, relu2):

    with open(fname, 'a+') as f:
        if first:
            if os.path.exists(fname): os.remove(fname)
            f.write('relu1,relu2\n')
        relu1 = get_saturation(relu1)
        relu2 = get_saturation(relu2)
        f.write('{},{}\n'.format(relu1, relu2))
    return relu1, relu2


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
