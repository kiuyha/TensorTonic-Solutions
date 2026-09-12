import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    X, y = np.array(X), np.array(y)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        
        if drop_last and len(X_batch) < batch_size:
            break
            
        yield X_batch, y_batch