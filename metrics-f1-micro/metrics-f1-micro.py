import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    maks = np.array(y_true) == np.array(y_pred)
    TP = np.sum(maks)
    FP = len(y_pred) - TP
    FN = len(y_true) - TP
    return (2 * TP) / (2 * TP + FP + FN)