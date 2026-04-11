import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    x_t, h_prev, h_t, W, U, b = cache
    dz = dh * (1 - np.pow(h_t, 2))
    dx_t = np.transpose(W) @ dz
    dh_prev = np.transpose(U) @ dz
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    return (dx_t, dh_prev, dW, dU, dz)
