from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numba import njit

from CONFIG import CONFIG


class CustomMetrics:
    """Factory class for creating custom evaluation metrics"""

    @staticmethod
    def comp_metric(
        predt: np.ndarray,
        rfr_data: np.ndarray,
        fwd_data: np.ndarray,
    ) -> Tuple[str, float]:
        position = np.clip(predt, CONFIG.MIN_INVESTMENT, CONFIG.MAX_INVESTMENT)

        N = len(rfr_data)
        strat_ret = rfr_data * (1 - position) + position * fwd_data
        excess_ret = strat_ret - rfr_data
        mean_excess = (1 + excess_ret).prod() ** (1 / N) - 1
        std = strat_ret.std()

        if std == 0:
            return "adj_sharpe", 0.0

        sharpe = mean_excess / std * np.sqrt(252)
        strat_vol = std * np.sqrt(252) * 100
        market_vol = fwd_data.std() * np.sqrt(252) * 100
        market_mean = (1 + fwd_data - rfr_data).prod() ** (1 / N) - 1

        vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2) if market_vol > 0 else 0
        return_penalty = 1 + ((max(0, (market_mean - mean_excess) * 100 * 252)) ** 2) / 100

        return "adj_sharpe", min(sharpe / (vol_penalty * return_penalty), 1e6)

    @staticmethod
    def create_volatility_adjusted_sharpe_xgb(
        rfr_data: np.ndarray,
        fwd_data: np.ndarray,
    ) -> Callable:
        """
        Create XGBoost custom metric with enclosed data.

        Args:
            rfr_data: Risk-free rate array
            fwd_data: Forward returns array

        Returns:
            Custom metric function for XGBoost
        """

        def metric(predt: np.ndarray, dtrain) -> Tuple[str, float]:
            position = np.clip(predt, CONFIG.MIN_INVESTMENT, CONFIG.MAX_INVESTMENT)

            N = len(rfr_data)
            strat_ret = rfr_data * (1 - position) + position * fwd_data
            excess_ret = strat_ret - rfr_data
            mean_excess = (1 + excess_ret).prod() ** (1 / N) - 1
            std = strat_ret.std()

            if std == 0:
                return "adj_sharpe", 0.0

            sharpe = mean_excess / std * np.sqrt(252)
            strat_vol = std * np.sqrt(252) * 100
            market_vol = fwd_data.std() * np.sqrt(252) * 100
            market_mean = (1 + fwd_data - rfr_data).prod() ** (1 / N) - 1

            vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2) if market_vol > 0 else 0
            return_penalty = 1 + ((max(0, (market_mean - mean_excess) * 100 * 252)) ** 2) / 100

            return "adj_sharpe", min(sharpe / (vol_penalty * return_penalty), 1e6)

        return metric

    @staticmethod
    def create_volatility_adjusted_sharpe_lgb(
        rfr_data: np.ndarray,
        fwd_data: np.ndarray,
    ) -> Callable:
        """
        Create LightGBM custom metric with enclosed data.

        Args:
            rfr_data: Risk-free rate array
            fwd_data: Forward returns array

        Returns:
            Custom metric function for LightGBM
        """

        def metric(preds: np.ndarray, train_data) -> Tuple[str, float, bool]:
            position = np.clip(preds, CONFIG.MIN_INVESTMENT, CONFIG.MAX_INVESTMENT)

            N = len(rfr_data)
            strat_ret = rfr_data * (1 - position) + position * fwd_data
            excess_ret = strat_ret - rfr_data
            mean_excess = (1 + excess_ret).prod() ** (1 / N) - 1
            std = strat_ret.std()

            if std == 0:
                return "adj_sharpe", 0.0, True

            sharpe = mean_excess / std * np.sqrt(252)
            strat_vol = std * np.sqrt(252) * 100
            market_vol = fwd_data.std() * np.sqrt(252) * 100
            market_mean = (1 + fwd_data - rfr_data).prod() ** (1 / N) - 1

            vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2) if market_vol > 0 else 0
            return_penalty = 1 + ((max(0, (market_mean - mean_excess) * 100 * 252)) ** 2) / 100

            adj_sharpe = min(sharpe / (vol_penalty * return_penalty), 1e6)
            return "adj_sharpe", adj_sharpe, True

        return metric
