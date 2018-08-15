import numpy as np
def get_stauration(x):
    """
    input:
        x: 2 dimensional numpy array, [batch, ]
    """
    ln = x.shape[0]
    ret = x[0, :].flatten() <= 0.
    for i in range(1, ln):
        ret = np.logical_and(ret, x[i, :].flatten() <= 0.)

    return ret/x[0, :].size
