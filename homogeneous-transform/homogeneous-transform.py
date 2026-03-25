import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    points = np.array(points)
    is_1d = points.ndim == 1
    if is_1d:
        points = points.reshape(1, -1)
    points_h = np.hstack((points, np.full((points.shape[0], 1), 1)))
    output = ((T @ points_h.T).T)[:, :-1]
    if is_1d:
        output = output.reshape(-1)
    return output
    