import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    X, y = np.array(X), np.array(y)
    class_sample = {
        k: round(v * test_size)
        for k, v in zip(*np.unique(y, return_counts=True))
    }
    rng = np.random.shuffle if rng is None else rng.shuffle
    
    train_idx, test_idx = [], []
    for k, v in class_sample.items():
        indices = np.where(y == k)[0]
        rng(indices)
        train_idx.append(np.sort(indices[v:]))
        test_idx.append(np.sort(indices[:v]))
        
    train_idx, test_idx = np.concatenate(train_idx), np.concatenate(test_idx)
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]