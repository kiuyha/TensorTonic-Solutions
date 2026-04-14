import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Fill NaN values in each feature column using column mean or median.
    """
    # Write code here
    X = np.array(X)
    with np.errstate(invalid='ignore'):
        if strategy == 'mean':
            stats = np.nanmean(X, axis=0)
        elif strategy == 'median':
            stats = np.nanmedian(X, axis=0)
    X[:, np.isnan(stats)] = 0.0
    return np.where(np.isnan(X), stats, X)
    