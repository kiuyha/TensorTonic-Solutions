import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here
    return math.exp(
        - sum(
            math.log(prob_distributions[i][actual_tokens[i]])
            for i in range(len(actual_tokens))
        ) / len(actual_tokens)
    )