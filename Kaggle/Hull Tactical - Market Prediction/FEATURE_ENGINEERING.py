import polars as pl
import polars.selectors as cs
import numpy as np
import itertools
from CONFIG import CONFIG
import torch
import time
from dataclasses import dataclass
from loguru import logger

# M* - Market Dynamics/Technical features.
# E* - Macro Economic features.
# I* - Interest Rate features.
# P* - Price/Valuation features.
# V* - Volatility features.
# S* - Sentiment features.
# MOM* - Momentum features.
# D* - Dummy/Binary features.


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result

    return wrapper


@dataclass(slots=True)
class FEATURE_ENGINEERING:
    df: pl.LazyFrame

    # ----------------------
    # Autocorrelation Features
    # ----------------------
    def _compute_autocorr_torch(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling autocorrelations for each asset using Torch and GPU acceleration.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe containing log returns

        Returns
        -------
        pl.LazyFrame
            Dataframe with autocorrelation features for multiple windows (10, 90, 252)
        """
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]
        windows = [10, 90, 252]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        data_np = df.collect().select(names).to_numpy().astype(np.float32)
        dates = df.collect().select(CONFIG.DATE_COL).to_series().to_list()
        data = torch.tensor(data_np, device=device)

        autocorrs = []
        for window in windows:
            rolling = data.unfold(0, window, 1).transpose(1, 2)
            mean = rolling.mean(dim=1, keepdim=True)
            centered = rolling - mean
            var = (centered**2).mean(dim=1)
            autocorr_num = (centered[:, 1:, :] * centered[:, :-1, :]).mean(dim=1)
            autocorr = autocorr_num / var
            output_dates = dates[window - 1 :]
            schema = [f"{name}_auto_corr_{window}" for name in names]
            autocorr_df = (
                pl.DataFrame(autocorr.detach().cpu().numpy(), schema=schema)
                .with_columns(pl.Series(CONFIG.DATE_COL, output_dates))
                .select([CONFIG.DATE_COL] + schema)
            )
            autocorrs.append(autocorr_df)

        all_auto_corrs = pl.DataFrame().with_columns(pl.Series(CONFIG.DATE_COL, dates))
        for autocorr in autocorrs:
            all_auto_corrs = all_auto_corrs.join(autocorr, how="left", on=CONFIG.DATE_COL)

        return all_auto_corrs.fill_null(strategy="backward").lazy()

    # ----------------------
    # Return LAG Features
    # ----------------------
    def _compute_lag(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling skewness of returns for each asset over windows [5, 10, 90, 252].

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with return features

        Returns
        -------
        pl.LazyFrame
            Dataframe with rolling skew features
        """
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL not in i]
        return (
            df.with_columns(
                [pl.col(col).shift(window).alias(f"{col}_lag_{window}") for col in names for window in [5, 10, 90, 252]]
            )
            .drop([i for i in df.collect_schema().names() if i != CONFIG.DATE_COL])
            .fill_null(0)
        )
        
        
    # ----------------------
    # Return Skew Features
    # ----------------------
    def _compute_return_skew(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling skewness of returns for each asset over windows [5, 10, 90, 252].

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with return features

        Returns
        -------
        pl.LazyFrame
            Dataframe with rolling skew features
        """
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL and CONFIG.BINARY_FEATURE not in i]
        return (
            df.with_columns(
                [pl.col(col).rolling_skew(window_size=window).alias(f"{col}_return_skew_{window}") for col in names for window in [5, 10, 90, 252]]
            )
            .drop([i for i in df.collect_schema().names() if i != CONFIG.DATE_COL])
            .fill_null(0)
        )

    # ----------------------
    # Rolling Statistics Features
    # ----------------------
    def _compute_rolling(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling mean, std, and SMA ratio for return features over windows [5, 10, 90, 252].

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with return features

        Returns
        -------
        pl.LazyFrame
            Dataframe with rolling statistics and SMA ratio features
        """
        df = df.select(
            [CONFIG.DATE_COL] + [col for col in self.df.collect_schema().names() if col != CONFIG.DATE_COL and CONFIG.BINARY_FEATURE not in col]
        )
        names = df.collect_schema().names()
        names = [i for i in names if i != CONFIG.DATE_COL]
        windows = [5, 10, 90, 252]
        return (
            df.with_columns(
                [
                    pl.col(col).rolling_mean(window_size=window, min_periods=2).alias(f"{col}_sma_{window}")
                    for col in names
                    if col
                    for window in windows
                ]
                + [
                    pl.col(col).rolling_std(window_size=window, min_periods=2).alias(f"{col}_vol_{window}")
                    for col in names
                    if col
                    for window in windows
                ]
                + [
                    (
                        pl.col(col).rolling_mean(window_size=window, min_periods=2)
                        / (pl.when(pl.col(col) < 0).then(pl.col(col)).otherwise(0.0).rolling_std(window_size=window, min_periods=2) + 1e-8)
                    ).alias(f"{col}_{window}_sortino")
                    for col in names
                    if col
                    for window in windows
                ]
            )
            .with_columns([(pl.col(col) / pl.col(f"{col}_sma_{window}")).alias(f"{col}_{window}_sharpe") for col in names for window in windows])
            .drop(names)
        )

    # ----------------------
    # Beta Features
    # ----------------------
    def _compute_betas(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling betas for all asset pairs using Polars rolling covariance.

        Beta of asset i with respect to asset j at time t is computed as:
            beta[i,j] = cov(i,j) / var(j)

        This method uses Polars rolling_cov and rolling_var without GPU acceleration.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe containing asset returns. Must include a date column defined in CONFIG.DATE_COL.

        Returns
        -------
        pl.LazyFrame
            Dataframe with rolling beta features for all asset pairs over a 90-day window.
            The output columns are named as "beta_{asset_i}_{asset_j}".
        """
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]

        # Generate all unique asset pairs
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        return (
            df.with_columns(
                [
                    pl.rolling_cov(a=pl.col(p1), b=pl.col(p2), window_size=90, min_periods=2).alias(f"beta_{p1}_{p2}")
                    / pl.col(p1).rolling_var(window_size=90, min_periods=2)
                    for p1, p2 in pairs
                    if p1 != CONFIG.DATE_COL or p2 != CONFIG.DATE_COL
                ]
            )
            .drop(names)
            .fill_null(0)
        )

    # ----------------------
    # Pair Features
    # ----------------------
    def _compute_pairs_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        names = df.collect_schema().names()
        names = [i for i in names if i != CONFIG.DATE_COL]
        err = 1e-6

        exprs = []
        for pair1, pair2 in itertools.combinations(names, 2):
            p1 = pl.col(f"{pair1}")
            p2 = pl.col(f"{pair2}")
            exprs.extend(
                [
                    # Polynomial
                    ((p1 + p2) ** 2).alias(f"{pair1}_{pair2}_poly2"),
                    ((p1 + p2) ** 3).alias(f"{pair1}_{pair2}_poly3"),
                    ((p1 - p2) ** 2).alias(f"{pair1}_{pair2}_diff_squared"),
                    # Nonlinear transforms
                    (p1 * p2).sqrt().alias(f"{pair1}_{pair2}_sqrt_mul"),
                    (1 + (p1 * p2)).log().alias(f"{pair1}_{pair2}_log_mul"),
                    (p1 - p2).exp().alias(f"{pair1}_{pair2}_exp_diff"),
                    # Statistical / Comparative
                    pl.when((p1 + p2) != 0).then(2 * p1 * p2 / (p1 + p2)).otherwise(None).alias(f"{pair1}_{pair2}_harmonic_mean"),
                    pl.when(p2 != 0).then((p1 / p2).arctan()).otherwise(None).alias(f"{pair1}_{pair2}_atan_ratio"),
                    pl.when(((p1 + err) / (p2 + err)) > 0).then(((p1 + err) / (p2 + err)).log()).otherwise(None).alias(f"{pair1}_{pair2}_log_ratio"),
                    # Logistic / sigmoid
                    (1 / (1 + (p1 - p2).neg().exp())).alias(f"{pair1}_{pair2}_sigmoid_diff"),
                ]
            )
        return df.with_columns(exprs).fill_null(0).fill_nan(0).drop(names)

    def create_market_features(self) -> pl.DataFrame:
        """
        Create all engineered market features including temporal, returns, lags,
        autocorrelation, OBV, skewness, volume z-score, market stats, ATR, and rolling stats.

        Returns
        -------
        pl.LazyFrame
            Fully feature-engineered dataframe ready for modeling
        """
        autocorr_df = self._compute_autocorr_torch(df=self.df)
        skew_df = self._compute_return_skew(df=self.df)
        lag_df = self._compute_lag(df=self.df)
        rolling_stats_df = self._compute_rolling(df=self.df)
        beta_df = self._compute_betas(df=self.df)
        interactions_df = self._compute_pairs_features(df=self.df)

        final_df = (
            self.df.join(autocorr_df, on=CONFIG.DATE_COL)
            .join(lag_df, on=CONFIG.DATE_COL)
            .join(skew_df, on=CONFIG.DATE_COL)
            .join(rolling_stats_df, on=CONFIG.DATE_COL)
            .join(beta_df, on=CONFIG.DATE_COL)
            .join(interactions_df, on=CONFIG.DATE_COL)
            .collect()
        )

        non_binary_cols = [col for col in final_df.columns if final_df[col].n_unique() > 3 and col != CONFIG.DATE_COL]

        windows = [10, 90, 252]

        final_df = (
            final_df.with_columns(
                *[
                    self.zscore(col=col, mean_window=mean_w, std_window=std_w)
                    for col in non_binary_cols
                    for (mean_w, std_w) in itertools.product(windows, repeat=2)
                ],
            )
            .drop(non_binary_cols)
            .fill_null(0)
            .fill_nan(0)
            .drop_nulls()
        )

        cols_to_drop = [col for col in final_df.select(cs.numeric()).columns if final_df[col].is_infinite().any()]
        return final_df.drop(cols_to_drop)

    def zscore(self, col: str, mean_window: int, std_window: int) -> pl.Expr:
        return (
            (pl.col(col) - pl.col(col).rolling_mean(window_size=mean_window, min_periods=2))
            / pl.col(col).rolling_std(window_size=std_window, min_periods=2)
        ).alias(f"{col}_std_{mean_window}_{std_window}")
