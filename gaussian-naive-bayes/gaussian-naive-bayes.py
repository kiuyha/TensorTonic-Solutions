import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train, y_train, X_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test)
    stat_c = {}
    classes = tuple(set(y_train))
    for c in classes:
        if c not in stat_c:
            stat_c[c] = {}
        rows = X_train[y_train == c]
        stat_c[c]['prior'] = len(rows) / len(X_train)
        stat_c[c]['mean'] = np.mean(rows, axis=0)
        stat_c[c]['var'] = np.std(rows, axis=0) ** 2 + 1e-9

    return [
        classes[np.argmax([
            np.log(stat['prior']) + (
                -0.5 * np.log(2 * np.pi * stat['var']) -
                ((x - stat['mean']) ** 2 / (2 * stat['var']))
            ).sum()
            for stat in stat_c.values()
        ])]
        for x in X_test
    ]