import numpy as np
from scipy.special import expit, logit


def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e
