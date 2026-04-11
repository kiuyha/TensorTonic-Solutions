def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    counts = {}
    for i in range(0, len(tokens) - 1):
        bigram = (tokens[i], tokens[i + 1])
        if bigram not in counts:
            counts[bigram] = 1
        else:
            counts[bigram] += 1
        
    vocabs = set(tokens)
    
    probs = {
        (w1, w2): ((counts.get((w1, w2), 0) + 1) / (
            sum(
                count
                for bigram, count in counts.items()
                if bigram[0] == w1
            ) + len(vocabs)
        ))
        for w2 in vocabs
        for w1 in vocabs
    }
    
    return counts, probs