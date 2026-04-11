import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X)
    if X.shape[0] < 2 or X.ndim != 2:
        return None
    X_centered = X - np.mean(X, axis=0)
    return X_centered.T @ X / (X.shape[0] - 1)