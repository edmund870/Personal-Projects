from dataclasses import dataclass
import polars as pl
import numpy as np
import torch
import re
import itertools
import time
from scipy.stats import rankdata, norm
from scipy.signal import hilbert
import pywt
from CONFIG import CONFIG


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
    """
    Feature engineering for financial time-series datasets.

    Computes temporal, market, and statistical features for trading assets
    using Polars LazyFrames and optionally GPU-accelerated Torch computations.

    Parameters
    ----------
    df : pl.LazyFrame
        Input dataframe with a date column specified by CONFIG.DATE_COL and asset columns.

    Attributes
    ----------
    START_INDEX : int
        Start index of dataset (default 0, e.g., Sept 11, 2017)
    END_INDEX : int
        End index of dataset (default 1916, e.g., April 18, 2025)
    TRADING_DAYS_PER_YEAR : int
        Approximate number of trading days per year (252)
    TRADING_DAYS_PER_QUARTER : int
        Approximate number of trading days per quarter (63)
    TRADING_DAYS_PER_MONTH : int
        Approximate number of trading days per month (21)
    TRADING_DAYS_PER_WEEK : int
        Approximate number of trading days per week (5)
    reference_points : dict
        Key reference indices for year-start points
    """

    df: pl.LazyFrame

    START_INDEX = 0
    END_INDEX = 1916
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_QUARTER = 63
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    reference_points = {
        "year_2018_start": 84,
        "year_2019_start": 336,
        "year_2020_start": 588,
        "year_2021_start": 840,
        "year_2022_start": 1092,
        "year_2023_start": 1344,
        "year_2024_start": 1596,
        "year_2025_start": 1848,
    }

    # ----------------------
    # Temporal Features
    # ----------------------
    def _cast_bool(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Cast all Boolean columns in the dataframe to Int8.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe

        Returns
        -------
        pl.LazyFrame
            Dataframe with Boolean columns cast to Int8
        """
        return df.with_columns(pl.col(col).cast(pl.Int8) for col, dtype in df.collect_schema().items() if dtype == pl.Boolean)

    def _basic_temporal_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute basic time-based features from the index or date column.

        Features include years since start, approximate year, day of trading year,
        quarter/month/week/day features, etc.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe

        Returns
        -------
        pl.LazyFrame
            Dataframe with additional temporal features
        """
        return df.with_columns(
            [
                (pl.col(CONFIG.DATE_COL) / self.TRADING_DAYS_PER_YEAR).alias("years_since_start"),
                (2017 + (pl.col(CONFIG.DATE_COL) / self.TRADING_DAYS_PER_YEAR).floor()).alias("approx_year"),
                (pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR).alias("day_of_trading_year"),
                ((pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_QUARTER).floor().alias("quarter_of_year"),
                ((pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_MONTH).floor().alias("month_of_year"),
                ((pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_WEEK).floor().alias("week_of_year"),
                (pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_WEEK).alias("day_of_week"),
                ((pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR) % self.TRADING_DAYS_PER_MONTH).alias("day_of_month"),
                ((pl.col(CONFIG.DATE_COL) % self.TRADING_DAYS_PER_YEAR) % self.TRADING_DAYS_PER_QUARTER).alias("day_of_quarter"),
            ]
        )

    def _cyclical_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Encode temporal features cyclically using sine and cosine transformations.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with temporal features

        Returns
        -------
        pl.LazyFrame
            Dataframe with cyclical encodings for annual, quarterly, monthly, and weekly cycles
        """
        return df.with_columns(
            [
                (2 * np.pi * pl.col("day_of_trading_year") / self.TRADING_DAYS_PER_YEAR).sin().alias("annual_sin"),
                (2 * np.pi * pl.col("day_of_trading_year") / self.TRADING_DAYS_PER_YEAR).cos().alias("annual_cos"),
                (2 * np.pi * pl.col("day_of_quarter") / self.TRADING_DAYS_PER_QUARTER).sin().alias("quarterly_sin"),
                (2 * np.pi * pl.col("day_of_quarter") / self.TRADING_DAYS_PER_QUARTER).cos().alias("quarterly_cos"),
                (2 * np.pi * pl.col("day_of_month") / self.TRADING_DAYS_PER_MONTH).sin().alias("monthly_sin"),
                (2 * np.pi * pl.col("day_of_month") / self.TRADING_DAYS_PER_MONTH).cos().alias("monthly_cos"),
                (2 * np.pi * pl.col("day_of_week") / self.TRADING_DAYS_PER_WEEK).sin().alias("weekly_sin"),
                (2 * np.pi * pl.col("day_of_week") / self.TRADING_DAYS_PER_WEEK).cos().alias("weekly_cos"),
                (2 * np.pi * pl.col("week_of_year") / 50).sin().alias("week_of_year_sin"),
                (2 * np.pi * pl.col("week_of_year") / 50).cos().alias("week_of_year_cos"),
            ]
        )

    def _market_calendar_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Add market-specific calendar features such as first/last trading day,
        options expiry, and triple witching.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with day/month/quarter features

        Returns
        -------
        pl.LazyFrame
            Dataframe with calendar effect features as Boolean columns
        """
        return df.with_columns(
            [
                (pl.col("day_of_month") == 0).alias("is_first_trading_day_month"),
                (pl.col("day_of_month") == 20).alias("is_last_trading_day_month"),
                (pl.col("day_of_quarter") == 0).alias("is_first_trading_day_quarter"),
                (pl.col("day_of_quarter") >= 60).alias("is_last_trading_day_quarter"),
                (pl.col("day_of_month") <= 4).alias("is_first_week_month"),
                (pl.col("day_of_month") >= 16).alias("is_last_week_month"),
                ((pl.col("day_of_month") >= 14) & (pl.col("day_of_month") <= 15) & (pl.col("day_of_week") == 4)).alias("is_options_expiry"),
                (
                    (pl.col("month_of_year").is_in([2, 5, 8, 11]))
                    & (pl.col("day_of_month") >= 14)
                    & (pl.col("day_of_month") <= 15)
                    & (pl.col("day_of_week") == 4)
                ).alias("is_triple_witching"),
            ]
        )

    def _create_temporal_features(self) -> pl.LazyFrame:
        """
        Wrapper to create all temporal and market calendar features, and cast Boolean columns.

        Returns
        -------
        pl.LazyFrame
            Dataframe with all temporal and cyclical features
        """
        return self.df.pipe(self._basic_temporal_features).pipe(self._market_calendar_features).pipe(self._cast_bool)

    # ----------------------
    # Return and Lag Features
    # ----------------------
    def _compute_returns(self) -> pl.LazyFrame:
        """
        Compute log returns for all columns in CONFIG.RETURNS_FEATURES.

        Returns
        -------
        pl.LazyFrame
            Dataframe with log returns and date column
        """
        names = [col for col in self.df.collect_schema().names() if any([i for i in CONFIG.RETURNS_FEATURES if i in col])]
        return (
            self.df.select([CONFIG.DATE_COL] + names)
            .with_columns([(pl.col(col) + 1).log().diff().alias(f"{col}_log_ret") for col in names])
            .drop(names)
            .fill_null(0)
        )

    def _compute_lag_returns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute lagged returns for previous 0-5 periods.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe containing log returns

        Returns
        -------
        pl.LazyFrame
            Dataframe with lagged return features
        """
        names = [col for col in df.collect_schema().names() if col != CONFIG.DATE_COL]
        return df.with_columns([pl.col(col).shift(i).alias(f"{col}_return_lag_{i}") for col in names for i in range(6)]).drop(names).fill_null(0)

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

    def _compute_betas_torch(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute rolling betas for all pairs of assets over multiple windows using Torch GPU acceleration.

        Beta of asset i with respect to asset j at time t is computed as:
            beta[i,j] = cov(i,j) / var(j)

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe containing asset returns. Must include a date column defined in CONFIG.DATE_COL.

        Returns
        -------
        pl.LazyFrame
            Dataframe with rolling beta features for all asset pairs and windows [10, 90, 252].
            The output columns are named as "{asset_i}_{asset_j}_{window}_beta".
        """
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]
        windows = [10, 90, 252]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Generate all unique asset pairs
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

        # Convert Polars dataframe to NumPy array, then Torch tensor on device
        data_np = df.collect().select(names).to_numpy().astype(np.float32)  # (T, N)
        dates = df.collect().select(CONFIG.DATE_COL).to_series().to_list()
        T, N = data_np.shape
        data = torch.tensor(data_np, device=device)

        betas = []

        for window in windows:
            # Create rolling windows
            rolling = data.unfold(0, window, 1)  # shape: (T_eff, window, N)
            T_eff = rolling.shape[0]
            rolling = rolling.transpose(1, 2)  # shape: (T_eff, N, window)

            # Center the data
            mean = rolling.mean(dim=2, keepdim=True)  # (T_eff, N, 1)
            centered = rolling - mean  # (T_eff, N, window)

            # Compute covariance: cov[i,j] for each window
            cov = torch.einsum("nij,nkj->nik", centered, centered) / (window - 1)  # (T_eff, N, N)

            # Variance of each asset (diagonal)
            var = torch.diagonal(cov, dim1=1, dim2=2)  # (T_eff, N)
            var_broadcast = var.unsqueeze(1).expand(-1, N, -1)  # (T_eff, N, N)

            # Compute beta with safe division
            beta = torch.where(var_broadcast == 0, torch.tensor(float("nan"), device=device), cov / var_broadcast)

            # Remove self-betas (diagonal)
            mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0).expand(T_eff, -1, -1)
            beta = beta.masked_fill(mask, float("nan"))

            # Extract beta values for the defined pairs
            beta_pairs = []
            for i_name, j_name in pairs:
                i_idx = names.index(i_name)
                j_idx = names.index(j_name)
                beta_pairs.append(beta[:, i_idx, j_idx])

            # Stack all pair betas
            beta_matrix = torch.stack(beta_pairs, dim=1)  # (T_eff, num_pairs)

            beta_cpu = beta_matrix.detach().cpu().numpy()
            output_dates = dates[window - 1 : window - 1 + T_eff]

            # Construct Polars DataFrame
            pair_labels = [f"{i}_{j}_{window}_beta" for i, j in pairs]
            beta_df = pl.DataFrame(beta_cpu, schema=pair_labels)
            beta_df = beta_df.with_columns(pl.Series(CONFIG.DATE_COL, output_dates)).select([CONFIG.DATE_COL] + pair_labels)

            betas.append(beta_df)

        # Merge all windows into single dataframe
        all_betas = pl.DataFrame().with_columns(pl.Series(CONFIG.DATE_COL, dates))
        for beta in betas:
            all_betas = all_betas.join(beta, how="left", on=CONFIG.DATE_COL)

        return all_betas.fill_null(strategy="backward").lazy()

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
    # On-Balance Volume Features
    # ----------------------
    def _compute_obv(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute On-Balance Volume (OBV) for all instruments over multiple windows.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with Volume/Close and adjusted Volume/Close columns

        Returns
        -------
        pl.LazyFrame
            Dataframe with OBV features over windows [5, 10, 90, 252]
        """
        df = df.select(
            [CONFIG.DATE_COL] + [col for col in df.collect_schema().names() if any(i in col for i in ["Volume", "adj_volume", "Close", "adj_close"])]
        )
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]

        instruments = set()
        for col in names:
            match = re.match(r"(.+)_(Volume|Close|adj_volume|adj_close)", col)
            if match:
                instruments.add(match.group(1))

        return (
            df.with_columns(
                [
                    pl.when(pl.col(f"{col}_Close") > pl.col(f"{col}_Close").shift())
                    .then(pl.col(f"{col}_Volume"))
                    .when(pl.col(f"{col}_Close") < pl.col(f"{col}_Close").shift())
                    .then(-pl.col(f"{col}_Volume"))
                    .otherwise(0)
                    .rolling_sum(window_size=window)
                    .alias(f"{col}_obv_{window}")
                    for col in instruments
                    if all(c in names for c in [f"{col}_Volume", f"{col}_Close"])
                    for window in [5, 10, 90, 252]
                    if any([f"{col}_obv_{window}" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
                + [
                    pl.when(pl.col(f"{col}_adj_close") > pl.col(f"{col}_adj_close").shift())
                    .then(pl.col(f"{col}_adj_volume"))
                    .when(pl.col(f"{col}_adj_close") < pl.col(f"{col}_adj_close").shift())
                    .then(-pl.col(f"{col}_adj_volume"))
                    .otherwise(0)
                    .rolling_sum(window_size=window)
                    .alias(f"{col}_obv_{window}")
                    for col in instruments
                    if all(c in names for c in [f"{col}_adj_volume", f"{col}_adj_close"])
                    for window in [5, 10, 90, 252]
                    if any([f"{col}_obv_{window}" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
            )
            .drop(names)
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
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]

        return (
            df.with_columns(
                [
                    pl.col(col).rolling_skew(window_size=window).alias(f"{col}_return_skew_{window}")
                    for col in names
                    for window in [5, 10, 90, 252]
                    if any([f"{col}_return_skew_{window}" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
            )
            .drop(names)
            .fill_null(0)
        )

    # ----------------------
    # Volume Z-score Features
    # ----------------------
    def _compute_volume_z(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute z-score of log-volume for each asset.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with Volume/adj_volume columns

        Returns
        -------
        pl.LazyFrame
            Dataframe with volume z-score features
        """
        df = df.select([CONFIG.DATE_COL] + [col for col in df.collect_schema().names() if any(i in col for i in ["Volume", "adj_volume"])])
        volume_arr = np.log(1 + np.nan_to_num(df.drop(CONFIG.DATE_COL).collect().to_numpy()))
        volume_z = ((volume_arr.T - volume_arr.mean(axis=1)) / (volume_arr.std(axis=1) + 1e-6)).T
        volume_z_df = (
            pl.DataFrame(volume_z, schema=[f"{col}_volume_z" for col in df.collect_schema().names()[1:]])
            .insert_column(0, df.select(CONFIG.DATE_COL).collect().to_series())
            .lazy()
        )
        return volume_z_df

    # ----------------------
    # Market Statistics Features
    # ----------------------
    def _compute_market_stats(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute market-wide statistics such as mean return, volatility, median,
        and market breadth measures.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with return features

        Returns
        -------
        pl.LazyFrame
            Dataframe with market-level statistics
        """
        market_arr = df.drop(CONFIG.DATE_COL).collect().to_numpy()
        markets_df = pl.DataFrame()

        for market in CONFIG.ASSETS.keys():
            names = df.collect_schema().names()
            names = [i for i in names if i.startswith(market)]
            if names.__len__() > 0:
                market_df = df.select(names).collect().to_numpy()
                markets_df = pl.concat(
                    [
                        markets_df,
                        pl.DataFrame(market_df.mean(axis=1), schema=[f"{market}_return"]),
                        pl.DataFrame(market_df.std(axis=1), schema=[f"{market}_vol"]),
                        pl.DataFrame(np.median(market_df, axis=1), schema=[f"{market}_median"]),
                        pl.DataFrame(np.where(market_df > 0, 1, 0).sum(axis=1) / market_df.shape[1], schema=[f"{market}_positive_breath"]),
                        pl.DataFrame(np.where(market_df < 0, 1, 0).sum(axis=1) / market_df.shape[1], schema=[f"{market}_negative_breath"]),
                    ],
                    how="horizontal",
                )

        return (
            pl.concat(
                [
                    markets_df,
                    pl.DataFrame(market_arr.mean(axis=1), schema=["market_return"]),
                    pl.DataFrame(market_arr.std(axis=1), schema=["market_vol"]),
                    pl.DataFrame(np.median(market_arr, axis=1), schema=["market_median"]),
                    pl.DataFrame(np.where(market_arr > 0, 1, 0).sum(axis=1) / market_arr.shape[1], schema=["positive_breath"]),
                    pl.DataFrame(np.where(market_arr < 0, 1, 0).sum(axis=1) / market_arr.shape[1], schema=["negative_breath"]),
                ],
                how="horizontal",
            )
            .insert_column(0, df.select(CONFIG.DATE_COL).collect().to_series())
            .lazy()
        )

    # ----------------------
    # Average True Range (ATR) Features
    # ----------------------
    def _compute_atr(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Compute Average True Range (ATR) over windows [5, 10, 90, 252] for each asset.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataframe with High, Low, Close and adjusted columns

        Returns
        -------
        pl.LazyFrame
            Dataframe with ATR features
        """
        df = df.select(
            [CONFIG.DATE_COL]
            + [col for col in df.collect_schema().names() if any(i in col for i in ["High", "Low", "Close", "adj_high", "adj_low", "adj_close"])]
        )
        names = [i for i in df.collect_schema().names() if i != CONFIG.DATE_COL]
        instruments = set()
        for col in df.collect_schema().names():
            match = re.match(r"(.+)_(High|Low|Close|adj_high|adj_low|adj_close)", col)
            if match:
                instruments.add(match.group(1))

        return (
            df.with_columns(
                [
                    pl.max_horizontal(
                        [
                            (pl.col(f"{col}_High") - pl.col(f"{col}_Low")).abs(),
                            (pl.col(f"{col}_High") - pl.col(f"{col}_Close").shift(1)).abs(),
                            (pl.col(f"{col}_Low") - pl.col(f"{col}_Close").shift(1)).abs(),
                        ]
                    )
                    .rolling_mean(window_size=window)
                    .alias(f"{col}_ATR_{window}")
                    for col in instruments
                    if all(c in names for c in [f"{col}_High", f"{col}_Low", f"{col}_Close"])
                    for window in [5, 10, 90, 252]
                ]
            )
            .drop(names)
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
        df = df.select([CONFIG.DATE_COL] + [col for col in self.df.collect_schema().names() if any(i in col for i in CONFIG.RETURNS_FEATURES)])
        names = df.collect_schema().names()
        names = [i for i in names if i != CONFIG.DATE_COL]
        windows = [5, 10, 90, 252]
        return (
            df.with_columns(
                [
                    pl.col(col).rolling_mean(window_size=window, min_periods=2).alias(f"{col}_sma_{window}")
                    for col in names
                    for window in windows
                    if (
                        any([f"{col}_sma_{window}" in impt_col for impt_col in CONFIG.IMPT_COL])
                        or any([f"{col}_{window}_sharpe" in impt_col for impt_col in CONFIG.IMPT_COL])
                    )
                ]
                + [
                    pl.col(col).rolling_std(window_size=window, min_periods=2).alias(f"{col}_vol_{window}")
                    for col in names
                    for window in windows
                    if any([f"{col}_vol_{window}" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
                + [
                    (
                        pl.col(col).rolling_mean(window_size=window, min_periods=2)
                        / (pl.when(pl.col(col) < 0).then(pl.col(col)).otherwise(0.0).rolling_std(window_size=window, min_periods=2) + 1e-8)
                    ).alias(f"{col}_{window}_sortino")
                    for col in names
                    for window in windows
                    if any([f"{col}_{window}_sortino" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
            )
            .with_columns(
                [
                    (pl.col(col) / pl.col(f"{col}_sma_{window}")).alias(f"{col}_{window}_sharpe")
                    for col in names
                    for window in windows
                    if any([f"{col}_{window}_sharpe" in impt_col for impt_col in CONFIG.IMPT_COL])
                ]
            )
            .drop(names)
        )

    def gaussian_rank_transform(self, arr: np.array):
        # n_samples, n_targets = arr.shape
        transformed_targets = np.full_like(arr, np.nan)

        for i, row in enumerate(arr):
            # Find valid (non-NaN) assets for this timestep
            valid_mask = ~np.isnan(row)
            valid_arr = row[valid_mask]
            ranks = rankdata(valid_arr, method="average")
            percentile_ranks = (ranks - 0.5) / (len(ranks))
            percentile_ranks = np.clip(percentile_ranks, 1e-8, 1 - 1e-8)
            gaussian_values = norm.ppf(percentile_ranks)
            transformed_targets[i, valid_mask] = gaussian_values
        return transformed_targets

    def _compute_pairs_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        names = df.collect_schema().names()
        names = [i for i in names if i != CONFIG.DATE_COL]
        err = 1e-6

        rank_df = df.collect()
        rank_arr = rank_df.drop(CONFIG.DATE_COL).to_numpy()
        ranked_arr = self.gaussian_rank_transform(rank_arr)
        ranked_df = pl.DataFrame(ranked_arr).insert_column(index=0, column=rank_df.select(CONFIG.DATE_COL).to_series())
        ranked_df.columns = rank_df.columns
        ranked_df = ranked_df.lazy()

        exprs = {"pairs": [], "ranked_pairs": []}

        # Prepare operations for each pair
        for pair1, pair2 in CONFIG.PAIRS:
            # Define p1 and p2 for each pair
            p1, p2 = pl.col(f"{pair1}_log_ret"), pl.col(f"{pair2}_log_ret")
            prefix = f"{pair1}_log_ret_{pair2}_log_ret"
            ranked_prefix = f"ranked_{prefix}"

            # Define all expressions for this pair
            operations = [
                ("add", p1 + p2),
                ("sub", p1 - p2),
                ("mul", p1 * p2),
                ("div", pl.when(p2 != 0).then(p1 / p2).otherwise(None)),
                ("poly2", (p1 + p2) ** 2),
                ("poly3", (p1 + p2) ** 3),
                ("diff_squared", (p1 - p2) ** 2),
                ("sqrt_mul", (p1 * p2).sqrt()),
                ("log_mul", (1 + (p1 * p2)).log()),
                ("exp_diff", (p1 - p2).exp()),
                ("min", p1.min()),
                ("max", p1.max()),
                ("mean", (p1 + p2) / 2),
                ("abs_diff", (p1 - p2).abs()),
                ("norm_diff", pl.when((p1 + p2) != 0).then((p1 - p2) / (p1 + p2 + err)).otherwise(None)),
                ("harmonic_mean", pl.when((p1 + p2) != 0).then(2 * p1 * p2 / (p1 + p2)).otherwise(None)),
                ("atan_ratio", pl.when(p2 != 0).then((p1 / p2).arctan()).otherwise(None)),
                ("log_ratio", pl.when(((p1 + err) / (p2 + err)) > 0).then(((p1 + err) / (p2 + err)).log()).otherwise(None)),
                ("sigmoid_diff", (1 / (1 + (p1 - p2).neg().exp()))),
                ("gt_flag", pl.when(p1 > p2).then(1).otherwise(0).cast(pl.Int8)),
            ]

            # Loop over operations and add to exprs if present in config
            for op_name, expr in operations:
                feature_name = f"{prefix}_{op_name}"
                if any([feature_name in impt_col for impt_col in CONFIG.IMPT_COL]):
                    exprs["pairs"].append(expr.alias(feature_name))

            # Similarly, for ranked pairs
            for op_name, expr in operations:
                feature_name = f"{ranked_prefix}_{op_name}"
                if any([feature_name in impt_col for impt_col in CONFIG.IMPT_COL]):
                    exprs["ranked_pairs"].append(expr.alias(feature_name))
        return (
            df.with_columns(exprs["pairs"])
            .fill_null(0)
            .fill_nan(0)
            .drop(names)
            .join(ranked_df.with_columns(exprs["ranked_pairs"]).fill_null(0).fill_nan(0).drop(names), on=CONFIG.DATE_COL)
        )

    def compute_hilbert_phase(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """
        Compute Hilbert transform phase for price series

        Args:
            prices: 1D array of prices
            window: smoothing window for phase calculation

        Returns:
            phase: instantaneous phase in radians
        """
        # Apply Hilbert transform
        analytic_signal = hilbert(prices, axis=0)
        phase = np.angle(analytic_signal)

        # Optional: smooth the phase to reduce noise
        if window > 1:
            phase = np.convolve(phase, np.ones(window) / window, mode="same")

        return phase

    def compute_wavelet_energy(self, prices: np.ndarray, wavelet: str = "db4", levels: int = 5) -> dict:
        """
        Compute wavelet energy at different frequency levels

        Args:
            prices: 1D array of prices
            wavelet: wavelet type (db4, haar, coif2, etc.)
            levels: number of decomposition levels

        Returns:
            dict with energy at each level
        """
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(prices, wavelet, level=levels, axis=0)

        # Compute energy at each level (sum of squares)
        energies = {}
        energies["approximation"] = np.sum(coeffs[0] ** 2, axis=0)

        for i, detail in enumerate(coeffs[1:], 1):
            energies[f"detail_level_{i}"] = np.sum(detail**2, axis=0)

        # Compute relative energies (normalized by total energy)
        total_energy = sum(energies.values())
        relative_energies = {f"{k}_rel": v / total_energy for k, v in energies.items()}

        return {**energies, **relative_energies}

    def add_hilbert_wavelet_features(
        self,
        hilbert_window: int = 90,
        wavelet_type: str = "db4",
        wavelet_levels: int = 4,
    ) -> pl.LazyFrame:
        """
        Add Hilbert phase and wavelet energy features to OHLCV dataframe

        Args:
            df: Polars DataFrame with OHLCV data
            price_cols: columns to compute features for
            group_col: column to group by (e.g., 'symbol' for multiple stocks)
            hilbert_window: smoothing window for Hilbert phase
            wavelet_type: type of wavelet to use
            wavelet_levels: number of wavelet decomposition levels
        """
        names = self.df.collect_schema().names()
        names = [i for i in names if i != CONFIG.DATE_COL]
        price_colS = [col for col in names if any(i in col for i in CONFIG.RETURNS_FEATURES)]

        for price_col in price_colS:
            prices = self.df.select(price_col).collect().to_numpy().flatten()

            # Skip if not enough data
            if len(prices) < max(hilbert_window * 2, 2**wavelet_levels):
                continue

            # Compute Hilbert phase features
            phase = self.compute_hilbert_phase(prices, hilbert_window)

            # Phase-based features
            phase_velocity = np.gradient(phase)  # Rate of phase change
            phase_acceleration = np.gradient(phase_velocity)  # Phase acceleration

            # Compute wavelet energy features
            wavelet_features = self.compute_wavelet_energy(prices, wavelet_type, wavelet_levels)

            # Add Hilbert features to dataframe
            result_df = self.df.with_columns(
                [
                    pl.Series(f"{price_col}_hilbert_phase", phase),
                    pl.Series(f"{price_col}_phase_velocity", phase_velocity),
                    pl.Series(f"{price_col}_phase_acceleration", phase_acceleration),
                    # Phase-based indicators
                    pl.Series(f"{price_col}_phase_cos", np.cos(phase)),
                    pl.Series(f"{price_col}_phase_sin", np.sin(phase)),
                ]
            ).drop(names)

            # Add wavelet energy features (these are scalars, so broadcast to all rows)
            for feature_name, feature_value in wavelet_features.items():
                result_df = result_df.with_columns([pl.lit(feature_value).alias(f"{price_col}_wavelet_{feature_name}")])

        return result_df

    # ----------------------
    # Final Market Feature Creation
    # ----------------------
    @timer
    def create_market_features(self) -> pl.DataFrame:
        """
        Create all engineered market features including temporal, returns, lags,
        autocorrelation, OBV, skewness, volume z-score, market stats, ATR, and rolling stats.

        Returns
        -------
        pl.LazyFrame
            Fully feature-engineered dataframe ready for modeling
        """
        temporal_df = self._create_temporal_features()
        returns_df = self._compute_returns()
        lags_df = self._compute_lag_returns(df=returns_df)
        # beta_df = self._compute_betas_torch(df=returns_df)
        autocorr_df = self._compute_autocorr_torch(df=returns_df)
        obv_df = self._compute_obv(df=self.df)
        skew_df = self._compute_return_skew(df=returns_df)
        # vol_z_df = self._compute_volume_z(df=self.df)
        market_stat_df = self._compute_market_stats(df=returns_df)
        # atr_df = self._compute_atr(df=self.df)
        rolling_stats_df = self._compute_rolling(df=self.df)
        interactions_df = self._compute_pairs_features(df=returns_df)
        hilbert_wave_df = self.add_hilbert_wavelet_features()

        final_df = (
            returns_df.join(lags_df, on=CONFIG.DATE_COL)
            # .join(beta_df, on=CONFIG.DATE_COL)
            .join(autocorr_df, on=CONFIG.DATE_COL)
            .join(obv_df, on=CONFIG.DATE_COL)
            .join(skew_df, on=CONFIG.DATE_COL)
            .join(temporal_df, on=CONFIG.DATE_COL)
            # .join(vol_z_df, on=CONFIG.DATE_COL)
            .join(market_stat_df, on=CONFIG.DATE_COL)
            # .join(atr_df, on=CONFIG.DATE_COL)
            .join(rolling_stats_df, on=CONFIG.DATE_COL)
            .join(interactions_df, on=CONFIG.DATE_COL)
            # .join(hilbert_wave_df, on=CONFIG.DATE_COL)
            .collect()
        )

        non_binary_cols = [col for col in final_df.columns if final_df[col].n_unique() > 3 and col != CONFIG.DATE_COL]

        windows = [10, 90, 252]

        return (
            final_df.with_columns(
                *[
                    self.zscore(col=col, mean_window=mean_w, std_window=std_w)
                    for col in non_binary_cols
                    for (mean_w, std_w) in itertools.product(windows, repeat=2)
                ],
            )
            .drop(non_binary_cols)
            .drop_nulls()
            .fill_nan(0)
        )

    def create_Y_market_features(self) -> pl.DataFrame:
        """
        Create all engineered market features including temporal, returns, lags,
        autocorrelation, OBV, skewness, volume z-score, market stats, ATR, and rolling stats.

        Returns
        -------
        pl.LazyFrame
            Fully feature-engineered dataframe ready for modeling
        """
        lags_df = self._compute_lag_returns(df=self.df)
        autocorr_df = self._compute_autocorr_torch(df=self.df)
        skew_df = self._compute_return_skew(df=self.df)
        market_stat_df = self._compute_market_stats(df=self.df)
        rolling_stats_df = self._compute_rolling(df=self.df)

        final_df = (
            self.df.join(lags_df, on=CONFIG.DATE_COL)
            # .join(beta_df, on=CONFIG.DATE_COL)
            .join(autocorr_df, on=CONFIG.DATE_COL)
            .join(skew_df, on=CONFIG.DATE_COL)
            .join(market_stat_df, on=CONFIG.DATE_COL)
            .join(rolling_stats_df, on=CONFIG.DATE_COL)
            .collect()
        )

        non_binary_cols = [col for col in final_df.columns if final_df[col].n_unique() > 3 and col != CONFIG.DATE_COL]

        windows = [10, 90, 252]

        return (
            final_df.with_columns(
                *[
                    self.zscore(col=col, mean_window=mean_w, std_window=std_w)
                    for col in non_binary_cols
                    for (mean_w, std_w) in itertools.product(windows, repeat=2)
                    if any([f"{col}_std_{mean_w}_{std_w}" in impt_col for impt_col in CONFIG.IMPT_COL])
                ],
            )
            .drop(non_binary_cols)
            .drop_nulls()
            .fill_nan(0)
        )

    def zscore(self, col: str, mean_window: int, std_window: int) -> pl.Expr:
        return (
            (pl.col(col) - pl.col(col).rolling_mean(window_size=mean_window, min_periods=2))
            / pl.col(col).rolling_std(window_size=std_window, min_periods=2)
        ).alias(f"{col}_std_{mean_window}_{std_window}")
