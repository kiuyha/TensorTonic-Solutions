import numpy as np

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
        if matrix.shape[0] != matrix.shape[1]:
            return None
    except:
        return None
    return np.linalg.eigvals(matrix)