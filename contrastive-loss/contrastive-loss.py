import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a, b, y =  np.array(a), np.array(b), np.array(y)
    d = np.linalg.norm(a - b, axis = 0 if a.ndim == 1 else 1)
    l = y * d**2 + (1-y) * np.maximum(0, margin-d)**2
    return np.sum(l) if reduction == 'sum' else np.mean(l) 