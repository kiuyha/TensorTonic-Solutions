import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    q = np.array(predictions)
    q[:] = epsilon / len(predictions)
    q[target] = (1 - epsilon) + epsilon / len(predictions)
    return - np.sum(q * np.log(predictions))