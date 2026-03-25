import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here
    v, w = np.array(v), np.array(w)
    
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan
        
    return np.arccos(
        np.clip(
            np.dot(v, w) / (norm_v * norm_w),
            -1, 1
        ),
    )