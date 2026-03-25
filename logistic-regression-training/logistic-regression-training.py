import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    weights = np.zeros(X.shape[1])
    bias = np.zeros(1)
    
    for i in range(steps):
        logits = _sigmoid(X @ weights + bias)
        weights = weights - lr * X.T @ (logits - y) / len(y)
        bias = bias - lr * sum(logits - y) / len(y)
        
    return weights, bias