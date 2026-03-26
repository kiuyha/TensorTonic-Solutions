import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    encoding = np.zeros((len(y_true), max(y_true) + 1))
    encoding[np.arange(len(y_true)), y_true] = 1
    return  - np.sum(encoding * np.log(y_pred)) / len(y_true)