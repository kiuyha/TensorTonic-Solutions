import numpy as np

def one_hot(y: list, num_classes=None) -> np.ndarray:
    """
    Returns a NumPy array with shape (N, K).
    """
    num_classes = num_classes if num_classes else max(y) + 1
    y = np.array(y)
    encoded = np.zeros((y.size, num_classes))
    encoded[np.arange(y.size), y] = 1.0
    return encoded