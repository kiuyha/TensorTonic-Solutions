import numpy as np
def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    p_t, y = np.array(predictions), np.array(targets)
    p_t[y == 0] = 1 - p_t[y == 0]
    return np.mean(
        -alpha * (1 - p_t) ** gamma * np.log(p_t)
    )