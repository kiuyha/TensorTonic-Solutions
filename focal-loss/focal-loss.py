import numpy as np

def focal_loss(p, y, gamma=2.0):
    """
    Compute Focal Loss for binary classification.
    """
    # Write code here
    p, y = np.array(p), np.array(y)
    return np.mean(
       - (np.power((1 - p), gamma) * y * np.log(p)) -
        (np.power(p, gamma) * (1 - y) * np.log(1-p))
    )