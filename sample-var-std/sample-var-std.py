import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    return np.var(x, ddof=1), np.std(x, ddof=1)