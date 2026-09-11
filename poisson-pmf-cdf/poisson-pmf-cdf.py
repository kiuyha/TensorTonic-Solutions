import math

def poisson_pmf_cdf(lam: float, k: int) -> dict:
    """
    Returns a dictionary with pmf and cdf.
    """
    # Write code here
    dists = []
    for i in range(k+1):
        dists.append(
            math.exp(-lam) * lam ** i / math.factorial(i)
        )
    return {
        "pmf": dists[-1],
        "cdf": sum(dists)
    }