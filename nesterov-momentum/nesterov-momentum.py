import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w, v, grad = np.array([w, v, grad])
    w_look = w - momentum * v
    new_v = momentum * v + lr * grad
    return w - new_v, new_v