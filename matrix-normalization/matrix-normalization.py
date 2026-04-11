import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    try:
        matrix = np.asarray(matrix)
        ord_norm_type = {
            'l1' : 1,
            'l2' : None,
            'max' : np.inf,
        }
        if matrix.ndim != 2 or norm_type not in ord_norm_type:
            return None
        return np.nan_to_num(matrix / np.linalg.norm(
          matrix,
          axis=axis,
          ord=ord_norm_type[norm_type],
          keepdims=True
        ))
    except:
        return None