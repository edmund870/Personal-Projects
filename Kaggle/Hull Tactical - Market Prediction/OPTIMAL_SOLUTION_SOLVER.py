import numpy as np
from numba import njit
from scipy.optimize import minimize, Bounds

MIN_INVESTMENT, MAX_INVESTMENT = 0, 2


class ParticipantVisibleError(Exception):
    pass


@njit
def ScoreMetric(submission, rfr, fwd, N, _):
    pos = submission
    if pos.max() > MAX_INVESTMENT:
        raise ParticipantVisibleError(f"Max {pos.max()} > {MAX_INVESTMENT}")
    if pos.min() < MIN_INVESTMENT:
        raise ParticipantVisibleError(f"Min {pos.min()} < {MIN_INVESTMENT}")

    strat_ret = rfr * (1 - pos) + pos * fwd

    excess_ret = strat_ret - rfr
    mean_excess = (1 + excess_ret).prod() ** (1 / N) - 1
    std = strat_ret.std()
    if std == 0:
        return 1e6

    sharpe = mean_excess / std * np.sqrt(252)
    strat_vol = std * np.sqrt(252) * 100

    market_vol = fwd.std() * np.sqrt(252) * 100
    market_mean = (1 + fwd - rfr).prod() ** (1 / N) - 1

    vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2) if market_vol > 0 else 0
    return_penalty = 1 + ((max(0, (market_mean - mean_excess) * 100 * 252)) ** 2) / 100

    return min(sharpe / (vol_penalty * return_penalty), 1e6)


def fun(x, rfr, fwd, N):
    """
    Optimization logic adapted from Hull Tactical Kaggle discussion #608349.
    """

    x = np.clip(x, 0, 2)
    return -ScoreMetric(x, rfr, fwd, N, "")


def optimize(train, rfr, fwd):
    x0 = np.full(len(train), 0.05)
    N = len(train)
    res = minimize(fun, x0, args=(rfr, fwd, N), method="Powell", bounds=Bounds(lb=0, ub=2), tol=1e-6)

    return res.x
