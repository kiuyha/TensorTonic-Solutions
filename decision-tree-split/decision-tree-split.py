import numpy as np

def _get_potential_thresholds(feature_values):
    """
    Get all the potential split points for a feature
    """
    sorted_unique_values = sorted(set(feature_values))

    # return midpoints between values
    return [
        (value + sorted_unique_values[idx - 1]) / 2

        for idx, value in enumerate(sorted_unique_values)

        # the element with idx 0 will not have element before it
        if idx != 0
    ]
def _split_data(X, y, feature_index, threshold):
    """
    Split data based on a threshold
    """
    X_left = []
    y_left = []
    X_right = []
    y_right = []
    for X, y in zip(X, y):
        if X[feature_index] <= threshold:
            X_left.append(X)
            y_left.append(y)
        else:
            X_right.append(X)
            y_right.append(y)

    return X_left, y_left, X_right, y_right
    
def _calculate_metric(y):
        """
        Calculate the Gini Score
        """
        unique_class = set(y)
        return 1 - sum([
            (y.count(class_name) / len(y) ) ** 2 # p^2
            
            for class_name in unique_class
        ])
    
def _calculate_score_split(y_left, y_right):
    """
    Calculate the weighted average for children nodes
    """
    n_left, n_right = len(y_left), len(y_right)
    n_total = n_left + n_right

    return ((n_left * _calculate_metric(y_left)) + (n_right * _calculate_metric(y_right))) / n_total

def _find_best_split(X, y):
    best_score_split = float('inf') # the score split should be the lowest so initialize with infinity
    best_split_tuple = None

    # make the order of features random
    feature_indices = list(range(len(X[0])))

    for feature_index in feature_indices:
        thresholds = _get_potential_thresholds(X[:, feature_index])
        for threshold in thresholds:
            _, y_left, _, y_right = _split_data(X, y, feature_index, threshold)

            score_split = _calculate_score_split(y_left, y_right)
            if score_split < best_score_split:
                best_score_split = score_split
                best_split_tuple = (
                    feature_index,
                    threshold,
                )
    return best_split_tuple
        
def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    return _find_best_split(np.asarray(X), np.asarray(y))