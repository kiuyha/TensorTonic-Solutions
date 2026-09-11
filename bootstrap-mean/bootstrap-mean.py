import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    rng = np.random.default_rng(seed)
    alpha = (1 - ci) / 2
    bootstrap_means = rng.choice(a=x, size=(n_bootstrap, len(x))).mean(axis=1)
    return {
        "bootstrap_mean": bootstrap_means.mean(),
        "lower": np.quantile(bootstrap_means, alpha),
        "upper": np.quantile(bootstrap_means, 1 - alpha)
    }