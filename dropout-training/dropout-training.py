import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    rng = rng.random if rng else np.random.random
    output = rng(size=x.shape)
    
    dropout_pattern = (output > p) * (1 / (1 - p))
    output = dropout_pattern * x
    
    return output, dropout_pattern