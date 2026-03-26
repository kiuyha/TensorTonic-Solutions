import numpy as np

def distance(x, y):
    x, y = np.array(x), np.array(y)
    return np.sum(np.power(x - y, 2), axis=0 if x.ndim == 1 else 1)

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Write code here
    return np.maximum(
        0,
        distance(anchor, positive) - distance(anchor, negative) + margin
    ).mean()