import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    K = num_classes if num_classes is not None else int(max(y_true)) + 1
    
    if K == 1:
        return None

    if y_true.size == 0 or y_pred.size == 0:
        return np.zeros((K, K))
        
    index  = y_true * K + y_pred
    matrix = np.bincount(index, minlength=K*K).reshape(K, K)
    if normalize == 'none':
        return matrix
        
    rows_sums = matrix.sum(
        axis=1 if normalize == 'true' else 0 if normalize == 'pred' else None,
        keepdims=True
    )
    rows_sums[rows_sums == 0] = 1
    return matrix / rows_sums