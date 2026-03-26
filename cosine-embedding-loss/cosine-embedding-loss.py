import numpy as np
def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) )
    
def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here
    if label  == 1:
        return 1 - cos_sim(x1, x2)
    else:
        return max(0, cos_sim(x1, x2) - margin)