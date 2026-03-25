import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    N = len(seqs)
    L = max_len if max_len else (max(len(seq) for seq in seqs) or 0)
    pad_arr = np.full((N, L), pad_value) 
    for i, seq in enumerate(seqs):
        max_idx = min(len(seq), L)
        pad_arr[i, :max_idx] = seq[:max_idx]
    return pad_arr
    