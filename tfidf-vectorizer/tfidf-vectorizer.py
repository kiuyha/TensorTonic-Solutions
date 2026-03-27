import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    vocabs = sorted(set((
        word
        for sentence in documents
        for word in sentence.lower().split()
    )))
    N = len(documents)
    result = np.zeros((N, len(vocabs)))
    counters = [
        Counter(sentence.lower().split())
        for sentence in documents
    ]
        
    for i, counter in enumerate(counters):
        total = sum(counter.values())
        for word, count in counter.items():
            tf = count / total
            df = sum(1 for c in counters if word in c)
            idf = math.log(N / df)
            result[i, vocabs.index(word)] = tf * idf

    return result, vocabs
    