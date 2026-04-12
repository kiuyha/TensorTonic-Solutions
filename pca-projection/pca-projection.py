import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X = np.array(X)
    X = X - X.mean(axis=0)
    matrix = X.T @ X / X.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_indices = sorted_indices[:k]
    top_components = eigenvectors[:, top_indices]
    return X @ top_components