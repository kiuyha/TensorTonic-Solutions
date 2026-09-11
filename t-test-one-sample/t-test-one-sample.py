import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    """
    Returns the t-statistic as a float.
    """
    n = len(x)
    x_mean = np.mean(x)
    s = np.sqrt(
        (1 / (n-1)) * np.sum(np.pow(x - x_mean, 2))
    )
    return float(
        (x_mean - mu0) / (s / np.sqrt(n)) 
    )