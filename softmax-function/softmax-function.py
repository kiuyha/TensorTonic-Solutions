import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.array(x)
    diff_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return diff_x / np.sum(diff_x, axis=-1, keepdims=True)
    