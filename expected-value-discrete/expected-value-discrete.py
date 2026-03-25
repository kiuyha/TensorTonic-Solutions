import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x, p = np.array(x), np.array(p)
    if np.sum(p) != 1:
        raise ValueError("Not sum to 1")
    return np.sum(x * p)
