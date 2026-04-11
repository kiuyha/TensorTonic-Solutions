import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here
    counters = [
        Counter(sentence)
        for sentence in docs
    ]
    vocabs = set(query_tokens)
    dfs = {
        word: sum(1 for c in counters if word in c)
        for word in vocabs
    }
    avg_doc_len = np.mean([len(doc) for doc in docs])
    result = []
    for i, counter in enumerate(counters):
        print(counter)
        values = []
        for word in vocabs:
            tf = counter.get(word, 0)
            idf = math.log(
                ((len(docs) - dfs[word] + 0.5) / (dfs[word] + 0.5)) + 1
            )
            values.append(
                idf * (
                    tf * (k1 + 1) /
                    (tf + k1 * (1 - b + b * (len(docs[i]) / avg_doc_len)))
                )
            )
        result.append(sum(values))
    return np.array(result)