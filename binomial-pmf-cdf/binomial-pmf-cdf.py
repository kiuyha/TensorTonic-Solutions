import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    pmf = comb(n, k) * p**k * (1 - p)**(n-k)
    k = np.arange(k + 1)
    cmf = np.sum(comb(n, k) * p**k * (1 - p)**(n-k))
    return pmf, cmf