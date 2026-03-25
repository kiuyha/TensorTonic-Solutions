import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros((seq_len, d_model))
    half = (d_model + 1) // 2
    val = np.arange(0, seq_len).reshape(-1, 1) / np.power(
        np.full((half, ), base), 
        np.arange(0, half) * 2 / d_model
    )
    pe[:, 0::2] = np.sin(val)
    pe[:, 1::2] = np.cos(val[:, :d_model // 2])
    return pe