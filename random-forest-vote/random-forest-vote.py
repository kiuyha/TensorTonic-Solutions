import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    x = np.array(predictions)
    return [
        max(
            zip(*np.unique(x[:, i], return_counts=True)),
            key=lambda x: x[1]
        )[0]
        for i in range(x.shape[1])
    ]