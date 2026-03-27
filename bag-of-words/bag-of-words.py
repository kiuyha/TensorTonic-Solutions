import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    dict_vocab = {
        word: idx
        for idx, word in enumerate(vocab)
    }
    result = np.zeros(len(vocab))
    for word in tokens:
        if word in dict_vocab:
            result[dict_vocab[word]] += 1
            
    return result.astype(int)