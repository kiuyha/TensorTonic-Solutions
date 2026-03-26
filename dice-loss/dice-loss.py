import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p, y = np.array(p), np.array(y)
    return 1 - (
        (2 * np.sum(p * y) + eps) /
        (np.sum(p) + np.sum(y) + eps)
    )