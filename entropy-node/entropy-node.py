import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    _, count  = np.unique(y, return_counts=True)
    pi = count / len(y)
    return -np.sum(pi * np.log2(pi))