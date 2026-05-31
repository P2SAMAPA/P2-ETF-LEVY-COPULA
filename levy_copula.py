import numpy as np
from scipy.stats import rankdata

def empirical_tail_dependence(returns, threshold=0.05):
    """
    Compute pairwise lower tail dependence coefficients for all ETFs.
    Returns a matrix (n x n) where entry (i,j) = P(rank_i <= p | rank_j <= p)
    where p = threshold (e.g., 0.05).
    """
    n_assets = returns.shape[1]
    # Convert to ranks
    ranks = np.apply_along_axis(rankdata, 0, returns.values)
    n = len(returns)
    p = threshold
    k = int(n * p)
    # Binary matrix: 1 if rank <= k
    tail_indicator = (ranks <= k).astype(int)
    # Tail dependence matrix
    tail_dep = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                tail_dep[i, j] = 1.0
            else:
                # P(rank_i <= k | rank_j <= k)
                both_tail = np.sum(tail_indicator[:, i] & tail_indicator[:, j])
                tail_given = np.sum(tail_indicator[:, j])
                if tail_given > 0:
                    tail_dep[i, j] = both_tail / tail_given
                else:
                    tail_dep[i, j] = 0.0
    return tail_dep

def levy_copula_scores(returns, threshold=0.05):
    """
    For each ETF, average tail dependence with all other ETFs.
    High score = tends to crash together with many others (systemic).
    Low score = crashes independently (diversifier).
    """
    tail_dep = empirical_tail_dependence(returns, threshold)
    n = len(tail_dep)
    scores = np.zeros(n)
    for i in range(n):
        # exclude self
        scores[i] = np.mean([tail_dep[i, j] for j in range(n) if j != i])
    tickers = returns.columns
    return {ticker: float(scores[i]) for i, ticker in enumerate(tickers)}
