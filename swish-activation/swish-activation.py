import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    return x / (1 + np.exp(-np.array(x)))