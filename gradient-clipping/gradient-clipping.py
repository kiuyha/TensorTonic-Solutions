import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    # Write code here
    g = np.asarray(g)
    norm_g = np.linalg.norm(g)
    if max_norm <= 0 or norm_g == 0:
        return g
    return g if norm_g <= max_norm else g * max_norm / norm_g