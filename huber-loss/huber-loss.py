import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    e = np.abs(y_true - y_pred)
    return np.mean(np.where(
        e <= delta,
        e ** 2 / 2,
        delta * (e - delta / 2)
    ))