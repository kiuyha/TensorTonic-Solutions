import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    std = np.std(X, axis=0)
    return np.cov(X.T, ddof=0) / np.outer(std, std)