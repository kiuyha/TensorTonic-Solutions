import math
def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    return [
        i 
        if i > 0
        else alpha * (math.exp(i) - 1)
        for i in x
    ]