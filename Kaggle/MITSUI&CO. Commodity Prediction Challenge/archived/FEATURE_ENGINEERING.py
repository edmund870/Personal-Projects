from dataclasses import dataclass
import polars as pl
import numpy as np
from CONFIG import CONFIG


@dataclass
class FEATURE_ENGINEERING:
    df: pl.LazyFrame

    # Key constants based on your dataset
    START_INDEX = 0  # Sept 11, 2017
    END_INDEX = 1916  # April 18, 2025

    # Trading calendar constants (approximate)
    TRADING_DAYS_PER_YEAR = 252
    TRADING_DAYS_PER_QUARTER = 63
    TRADING_DAYS_PER_MONTH = 21
    TRADING_DAYS_PER_WEEK = 5

    # Key reference points in your dataset
    reference_points = {
        "year_2018_start": 84,  # Approx Jan 1, 2018
        "year_2019_start": 336,  # Approx Jan 1, 2019
        "year_2020_start": 588,  # Approx Jan 1, 2020
        "year_2021_start": 840,  # Approx Jan 1, 2021
        "year_2022_start": 1092,  # Approx Jan 1, 2022
        "year_2023_start": 1344,  # Approx Jan 1, 2023
        "year_2024_start": 1596,  # Approx Jan 1, 2024
        "year_2025_start": 1848,  # Approx Jan 1, 2025
    }

    def cast_bool(self, df: pl.LazyFrame):
        return df.with_columns(pl.col(col).cast(pl.Int8) for col, dtype in df.collect_schema().items() if dtype == pl.Boolean)

    def _basic_temporal_features(self, df: pl.LazyFrame, index_col: str) -> pl.LazyFrame:
        """Basic time-based features from index"""
        return df.with_columns(
            [
                # Years since start (continuous)
                (pl.col(index_col) / self.TRADING_DAYS_PER_YEAR).alias("years_since_start"),
                # Approximate year (discrete)
                (2017 + (pl.col(index_col) / self.TRADING_DAYS_PER_YEAR).floor()).alias("approx_year"),
                # Days within year (0-251)
                (pl.col(index_col) % self.TRADING_DAYS_PER_YEAR).alias("day_of_trading_year"),
                # Quarter within year (0-3)
                ((pl.col(index_col) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_QUARTER).floor().alias("quarter_of_year"),
                # Month within year (0-11) - approximate
                ((pl.col(index_col) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_MONTH).floor().alias("month_of_year"),
                # Week within year (0-50)
                ((pl.col(index_col) % self.TRADING_DAYS_PER_YEAR) / self.TRADING_DAYS_PER_WEEK).floor().alias("week_of_year"),
                # Day of week (0-4, Mon-Fri)
                (pl.col(index_col) % self.TRADING_DAYS_PER_WEEK).alias("day_of_week"),
                # Day within month (0-20)
                ((pl.col(index_col) % self.TRADING_DAYS_PER_YEAR) % self.TRADING_DAYS_PER_MONTH).alias("day_of_month"),
                # Day within quarter (0-62)
                ((pl.col(index_col) % self.TRADING_DAYS_PER_YEAR) % self.TRADING_DAYS_PER_QUARTER).alias("day_of_quarter"),
            ]
        )

    def _cyclical_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Cyclical encoding of temporal features"""
        return df.with_columns(
            [
                # Annual cycle
                (2 * np.pi * pl.col("day_of_trading_year") / self.TRADING_DAYS_PER_YEAR).sin().alias("annual_sin"),
                (2 * np.pi * pl.col("day_of_trading_year") / self.TRADING_DAYS_PER_YEAR).cos().alias("annual_cos"),
                # Quarterly cycle
                (2 * np.pi * pl.col("day_of_quarter") / self.TRADING_DAYS_PER_QUARTER).sin().alias("quarterly_sin"),
                (2 * np.pi * pl.col("day_of_quarter") / self.TRADING_DAYS_PER_QUARTER).cos().alias("quarterly_cos"),
                # Monthly cycle
                (2 * np.pi * pl.col("day_of_month") / self.TRADING_DAYS_PER_MONTH).sin().alias("monthly_sin"),
                (2 * np.pi * pl.col("day_of_month") / self.TRADING_DAYS_PER_MONTH).cos().alias("monthly_cos"),
                # Weekly cycle
                (2 * np.pi * pl.col("day_of_week") / self.TRADING_DAYS_PER_WEEK).sin().alias("weekly_sin"),
                (2 * np.pi * pl.col("day_of_week") / self.TRADING_DAYS_PER_WEEK).cos().alias("weekly_cos"),
                # Week of year cycle
                (2 * np.pi * pl.col("week_of_year") / 50).sin().alias("week_of_year_sin"),
                (2 * np.pi * pl.col("week_of_year") / 50).cos().alias("week_of_year_cos"),
            ]
        )

    def _market_calendar_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Market-specific calendar effects"""
        return df.with_columns(
            [
                # First/Last trading day of month
                (pl.col("day_of_month") == 0).alias("is_first_trading_day_month"),
                (pl.col("day_of_month") == 20).alias("is_last_trading_day_month"),
                # First/Last trading day of quarter
                (pl.col("day_of_quarter") == 0).alias("is_first_trading_day_quarter"),
                (pl.col("day_of_quarter") >= 60).alias("is_last_trading_day_quarter"),
                # First/Last week of month
                (pl.col("day_of_month") <= 4).alias("is_first_week_month"),
                (pl.col("day_of_month") >= 16).alias("is_last_week_month"),
                # Options expiry (3rd Friday of month - approximately day 14-15 of trading month)
                ((pl.col("day_of_month") >= 14) & (pl.col("day_of_month") <= 15) & (pl.col("day_of_week") == 4)).alias("is_options_expiry"),
                # Triple witching (March, June, September, December - quarters 0,1,2,3 with options expiry)
                (
                    (pl.col("month_of_year").is_in([2, 5, 8, 11]))
                    & (pl.col("day_of_month") >= 14)
                    & (pl.col("day_of_month") <= 15)
                    & (pl.col("day_of_week") == 4)
                ).alias("is_triple_witching"),
            ]
        )

    def _seasonal_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Seasonal market patterns"""
        return df.with_columns(
            [
                # January effect (month 0)
                (pl.col("month_of_year") == 0).alias("is_january_effect"),
                # Sell in May (months 4-8: May through September)
                (pl.col("month_of_year").is_in([4, 5, 6, 7, 8])).alias("is_sell_in_may"),
                # Summer months (June, July, August - months 5,6,7)
                (pl.col("month_of_year").is_in([5, 6, 7])).alias("is_summer_months"),
                # Q4 rally (October, November, December - months 9,10,11)
                (pl.col("month_of_year").is_in([9, 10, 11])).alias("is_q4_rally"),
                # Turn of year effect (last 5 days of year + first 5 days)
                ((pl.col("day_of_trading_year") >= 247) | (pl.col("day_of_trading_year") <= 5)).alias("is_turn_of_year"),
                # Santa Claus rally (last 5 trading days of year)
                (pl.col("day_of_trading_year") >= 247).alias("is_santa_claus_rally"),
            ]
        )

    def _trading_session_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Trading session characteristics"""
        return df.with_columns(
            [
                # Day of week indicators
                (pl.col("day_of_week") == 0).alias("is_monday"),
                (pl.col("day_of_week") == 1).alias("is_tuesday"),
                (pl.col("day_of_week") == 2).alias("is_wednesday"),
                (pl.col("day_of_week") == 3).alias("is_thursday"),
                (pl.col("day_of_week") == 4).alias("is_friday"),
                # Week position
                (pl.col("day_of_week") == 0).alias("is_start_of_week"),
                (pl.col("day_of_week") == 4).alias("is_end_of_week"),
                (pl.col("day_of_week").is_in([1, 2, 3])).alias("is_mid_week"),
            ]
        )

    def _business_cycle_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Business and earnings cycle features"""
        return df.with_columns(
            [
                # Earnings seasons (roughly months 0,3,6,9 - Jan,Apr,Jul,Oct)
                (pl.col("month_of_year").is_in([0, 3, 6, 9])).alias("is_earnings_season"),
                # Fed meeting schedule (every ~6 weeks, 8 times per year)
                # Approximate: weeks 6,12,18,24,30,36,42,48 of the year
                (pl.col("week_of_year").is_in([6, 12, 18, 24, 30, 36, 42, 48])).alias("is_potential_fed_week"),
                # NFP week (first Friday of month - approximately first week)
                ((pl.col("day_of_month") <= 4) & (pl.col("day_of_week") == 4)).alias("is_nfp_week"),
                # CPI week (mid-month, approximately days 10-15)
                ((pl.col("day_of_month") >= 10) & (pl.col("day_of_month") <= 15)).alias("is_cpi_week"),
            ]
        )

    def create_market_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate comprehensive features for time series forecasting.

        Expected input DataFrame structure:
        - dateid: date identifier (1927 unique values)
        - ticker: ticker symbol (143 unique values per date)
        - open, high, low, close, volume: OHLCV data

        Returns DataFrame with original data plus engineered features.
        """

        # Ensure proper sorting for time series operations
        df = df.sort(["instr", CONFIG.DATE_COL])

        # ================================
        # BASE TRANSFORMATIONS (computed once)
        # ================================

        df = df.with_columns(
            [
                # Basic price relationships
                ((pl.col("High") - pl.col("Low")) / pl.col("Close")).alias("hl_range_pct"),
                ((pl.col("Close") - pl.col("Open")) / pl.col("Open")).alias("oc_return"),
                ((pl.col("High") - pl.col("Close")) / pl.col("Close")).alias("upper_shadow_pct"),
                ((pl.col("Close") - pl.col("Low")) / pl.col("Close")).alias("lower_shadow_pct"),
                (pl.col("High") - pl.col("Low")).alias("true_range_raw"),
                # Log prices for stability
                pl.col("Open").log().alias("log_open"),
                pl.col("High").log().alias("log_high"),
                pl.col("Low").log().alias("log_low"),
                pl.col("Close").log().alias("log_close"),
                (pl.col("Volume") + 1).log().alias("log_volume"),
                # Typical price and weighted close
                ((pl.col("High") + pl.col("Low") + pl.col("Close")) / 3).alias("typical_price"),
                ((pl.col("High") + pl.col("Low") + 2 * pl.col("Close")) / 4).alias("weighted_close"),
                # Dollar volume (used in multiple places)
                (pl.col("Volume") * pl.col("Close")).alias("dollar_volume"),
                # Basic returns (computed once, used everywhere)
                pl.col("Close").pct_change().over("instr").alias("return_1d"),
                pl.col("Close").log().diff().over("instr").alias("log_return_1d"),
            ]
        )

        # ================================
        # MULTI-TIMEFRAME RETURNS (computed once)
        # ================================

        windows = [2, 3, 5, 10, 20, 60]
        return_features = []

        for window in windows:
            return_features.extend(
                [
                    pl.col("Close").pct_change(window).over("instr").alias(f"return_{window}d"),
                    pl.col("log_close").diff(window).over("instr").alias(f"log_return_{window}d"),
                    pl.col("log_volume").diff(window).over("instr").alias(f"volume_change_{window}d"),
                    pl.col("typical_price").pct_change(window).over("instr").alias(f"typical_return_{window}d"),
                ]
            )

        df = df.with_columns(return_features)

        # ================================
        # LAGGED FEATURES (computed once)
        # ================================

        lag_windows = [1, 2, 3, 5, 10, 20, 60]
        lagged_features = []

        for window in lag_windows:
            lagged_features.extend(
                [
                    pl.col("Close").shift(window).over("instr").alias(f"close_lag_{window}"),
                    pl.col("Volume").shift(window).over("instr").alias(f"volume_lag_{window}"),
                    pl.col("log_return_1d").shift(window).over("instr").alias(f"log_return_1d_lag_{window}"),
                    pl.col("return_1d").shift(window).over("instr").alias(f"return_1d_lag_{window}"),
                ]
            )

        df = df.with_columns(lagged_features)

        # ================================
        # MOVING AVERAGES (computed once)
        # ================================

        ma_windows = [5, 10, 20, 50, 100]
        ma_features = []

        for window in ma_windows:
            ma_features.extend(
                [
                    pl.col("Close").rolling_mean(window).over("instr").alias(f"sma_{window}"),
                    pl.col("Volume").rolling_mean(window).over("instr").alias(f"volume_sma_{window}"),
                    pl.col("log_close").rolling_mean(window).over("instr").alias(f"log_sma_{window}"),
                    pl.col("Close").ewm_mean(span=window).over("instr").alias(f"ema_{window}"),
                ]
            )

        df = df.with_columns(ma_features)

        # Price relative to moving averages
        ma_ratio_features = []
        for window in ma_windows:
            ma_ratio_features.extend(
                [
                    (pl.col("Close") / pl.col(f"sma_{window}")).alias(f"close_sma_{window}_ratio"),
                    (pl.col("Close") / pl.col(f"ema_{window}")).alias(f"close_ema_{window}_ratio"),
                ]
            )

        df = df.with_columns(ma_ratio_features)

        # ================================
        # VOLATILITY FEATURES (computed once)
        # ================================

        volatility_windows = [5, 10, 20, 40, 60]
        volatility_features = []

        for window in volatility_windows:
            volatility_features.extend(
                [
                    # Historical volatility
                    pl.col("log_return_1d").rolling_std(window).over("instr").alias(f"volatility_{window}d"),
                    pl.col("return_1d").rolling_std(window).over("instr").alias(f"return_std_{window}d"),
                    pl.col("return_1d").rolling_var(window).over("instr").alias(f"return_variance_{window}d"),
                    # Average True Range
                    pl.col("true_range_raw").rolling_mean(window).over("instr").alias(f"atr_{window}"),
                    # High-Low volatility
                    pl.col("hl_range_pct").rolling_mean(window).over("instr").alias(f"hl_volatility_{window}"),
                    # Return range (max - min)
                    (pl.col("log_return_1d").rolling_max(window).over("instr") - pl.col("log_return_1d").rolling_min(window).over("instr")).alias(
                        f"return_range_{window}d"
                    ),
                ]
            )

        df = df.with_columns(volatility_features)

        # ================================
        # STATISTICAL MOMENTS AND QUANTILES (computed once)
        # ================================

        moment_windows = [10, 20, 40, 60]
        moment_features = []

        for window in moment_windows:
            moment_features.extend(
                [
                    # Higher moments
                    pl.col("return_1d").rolling_skew(window).over("instr").alias(f"return_skew_{window}d"),
                    # Distribution quantiles
                    pl.col("return_1d").rolling_quantile(quantile=0.01, window_size=window).over("instr").alias(f"return_q01_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.05, window_size=window).over("instr").alias(f"return_q05_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.10, window_size=window).over("instr").alias(f"return_q10_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.25, window_size=window).over("instr").alias(f"return_q25_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.75, window_size=window).over("instr").alias(f"return_q75_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.90, window_size=window).over("instr").alias(f"return_q90_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.95, window_size=window).over("instr").alias(f"return_q95_{window}d"),
                    pl.col("return_1d").rolling_quantile(quantile=0.99, window_size=window).over("instr").alias(f"return_q99_{window}d"),
                    # Rolling median and summary stats
                    pl.col("return_1d").rolling_median(window).over("instr").alias(f"return_median_{window}d"),
                    pl.col("return_1d").rolling_min(window).over("instr").alias(f"return_min_{window}d"),
                    pl.col("return_1d").rolling_max(window).over("instr").alias(f"return_max_{window}d"),
                    # Price extremes
                    pl.col("Close").rolling_min(window).over("instr").alias(f"close_min_{window}d"),
                    pl.col("Close").rolling_max(window).over("instr").alias(f"close_max_{window}d"),
                ]
            )

        df = df.with_columns(moment_features)

        # ================================
        # CROSS-SECTIONAL RANKINGS (computed once)
        # ================================

        ranking_features = [
            # Basic rankings
            pl.col("Close").rank().over(CONFIG.DATE_COL).alias("close_rank"),
            pl.col("Volume").rank().over(CONFIG.DATE_COL).alias("volume_rank"),
            pl.col("dollar_volume").rank().over(CONFIG.DATE_COL).alias("dollar_volume_rank"),
            pl.col("return_1d").rank().over(CONFIG.DATE_COL).alias("return_1d_rank"),
            pl.col("return_5d").rank().over(CONFIG.DATE_COL).alias("return_5d_rank"),
            pl.col("return_20d").rank().over(CONFIG.DATE_COL).alias("return_20d_rank"),
            pl.col("return_60d").rank().over(CONFIG.DATE_COL).alias("return_60d_rank"),
            pl.col("volatility_20d").rank().over(CONFIG.DATE_COL).alias("volatility_20d_rank"),
            pl.col("volatility_60d").rank().over(CONFIG.DATE_COL).alias("volatility_60d_rank"),
            # Momentum rankings
            (pl.col("Close") / pl.col("close_lag_20")).rank().over(CONFIG.DATE_COL).alias("momentum_20d_rank"),
            (pl.col("Close") / pl.col("close_lag_60")).rank().over(CONFIG.DATE_COL).alias("momentum_60d_rank"),
            # Mean reversion rankings
            pl.col("close_sma_20_ratio").rank().over(CONFIG.DATE_COL).alias("mean_reversion_20d_rank"),
            pl.col("close_sma_50_ratio").rank().over(CONFIG.DATE_COL).alias("mean_reversion_50d_rank"),
        ]

        df = df.with_columns(ranking_features)

        # ================================
        # CROSS-SECTIONAL STATISTICS (computed once)
        # ================================

        cross_sectional_features = [
            # Percentiles
            (pl.col("return_1d_rank") / pl.col("return_1d_rank").count().over(CONFIG.DATE_COL)).alias("return_1d_percentile"),
            (pl.col("volume_rank") / pl.col("volume_rank").count().over(CONFIG.DATE_COL)).alias("volume_percentile"),
            (pl.col("volatility_20d_rank") / pl.col("volatility_20d_rank").count().over(CONFIG.DATE_COL)).alias("volatility_percentile"),
            # Market-wide statistics
            pl.col("return_1d").mean().over(CONFIG.DATE_COL).alias("market_return"),
            pl.col("return_1d").std().over(CONFIG.DATE_COL).alias("market_volatility"),
            pl.col("return_1d").median().over(CONFIG.DATE_COL).alias("market_median_return"),
            pl.col("return_5d").std().over(CONFIG.DATE_COL).alias("market_volatility_5d"),
            # Cross-sectional dispersion
            (pl.col("return_1d").quantile(0.9).over(CONFIG.DATE_COL) - pl.col("return_1d").quantile(0.1).over(CONFIG.DATE_COL)).alias(
                "market_return_dispersion"
            ),
            (pl.col("return_1d").quantile(0.75).over(CONFIG.DATE_COL) - pl.col("return_1d").quantile(0.25).over(CONFIG.DATE_COL)).alias("market_iqr"),
            # Market breadth
            (pl.col("return_1d") > 0).mean().over(CONFIG.DATE_COL).alias("market_breadth"),
            (pl.col("return_5d") > 0).mean().over(CONFIG.DATE_COL).alias("market_breadth_5d"),
            # Excess returns vs market
            (pl.col("return_1d") - pl.col("return_1d").median().over(CONFIG.DATE_COL)).alias("excess_return_1d"),
            (pl.col("return_5d") - pl.col("return_5d").median().over(CONFIG.DATE_COL)).alias("excess_return_5d"),
            (pl.col("return_20d") - pl.col("return_20d").median().over(CONFIG.DATE_COL)).alias("excess_return_20d"),
            # Z-scores
            ((pl.col("return_1d") - pl.col("return_1d").mean().over(CONFIG.DATE_COL)) / pl.col("return_1d").std().over(CONFIG.DATE_COL)).alias(
                "return_1d_zscore"
            ),
            ((pl.col("log_volume") - pl.col("log_volume").median().over(CONFIG.DATE_COL)) / pl.col("log_volume").std().over(CONFIG.DATE_COL)).alias(
                "volume_zscore"
            ),
        ]

        df = df.with_columns(cross_sectional_features)

        # ================================
        # MICROSTRUCTURE FEATURES
        # ================================

        microstructure_features = [
            # Spread and impact measures
            ((pl.col("High") - pl.col("Low")) / pl.col("log_volume")).alias("spread_volume_ratio"),
            (pl.col("return_1d") / pl.col("log_volume")).alias("price_impact"),
            (pl.col("return_1d") / pl.col("log_volume").diff(1)).alias("volume_price_elasticity"),
            # Intraday components
            ((pl.col("Open") - pl.col("close_lag_1")) / pl.col("close_lag_1")).alias("overnight_return"),
            pl.col("oc_return").alias("intraday_return"),
            # Trading intensity
            (pl.col("Volume") / pl.col("true_range_raw")).alias("volume_per_price_range"),
            # Illiquidity measures
            (abs(pl.col("return_1d")) / (pl.col("dollar_volume") + 1e-8)).alias("amihud_illiquidity"),
            # Order flow proxies
            ((pl.col("Close") - pl.col("Low")) - (pl.col("High") - pl.col("Close"))) / pl.col("true_range_raw").alias("order_flow_imbalance"),
            # Market depth proxy
            (pl.col("Volume") / abs(pl.col("return_1d"))).alias("market_depth_proxy"),
            # Tick-level proxies
            (1 / pl.col("Close")).alias("relative_tick_size"),
            (pl.col("Close") % 0.05).alias("price_clustering_nickel"),
            (pl.col("Close") % 0.25).alias("price_clustering_quarter"),
            # Trade direction proxies
            pl.when(pl.col("Close") > (pl.col("High") + pl.col("Low")) / 2).then(1).otherwise(-1).alias("trade_direction_proxy"),
            pl.when(pl.col("Close") > pl.col("close_lag_1"))
            .then(1)
            .when(pl.col("Close") < pl.col("close_lag_1"))
            .then(-1)
            .otherwise(0)
            .alias("tick_direction"),
        ]

        df = df.with_columns(microstructure_features)

        # ================================
        # MOMENTUM AND TECHNICAL INDICATORS
        # ================================

        # RSI calculation
        rsi_windows = [5, 14, 30]
        rsi_features = []
        for window in rsi_windows:
            rsi_features.extend(
                [
                    pl.when(pl.col("log_return_1d") > 0)
                    .then(pl.col("log_return_1d"))
                    .otherwise(0)
                    .rolling_mean(window)
                    .over("instr")
                    .alias(f"avg_gain_{window}"),
                    pl.when(pl.col("log_return_1d") < 0)
                    .then(-pl.col("log_return_1d"))
                    .otherwise(0)
                    .rolling_mean(window)
                    .over("instr")
                    .alias(f"avg_loss_{window}"),
                ]
            )

        df = df.with_columns(rsi_features)

        # Complete RSI calculation
        for window in rsi_windows:
            df = df.with_columns([(100 - (100 / (1 + pl.col(f"avg_gain_{window}") / pl.col(f"avg_loss_{window}")))).alias(f"rsi_{window}")])

        # MACD
        macd_features = [
            (pl.col("Close").rolling_mean(window_size=12) - pl.col("Close").rolling_mean(window_size=26)).alias("macd_line"),
        ]
        df = df.with_columns(macd_features)

        df = df.with_columns(
            [
                pl.col("macd_line").ewm_mean(span=9).over("instr").alias("macd_signal"),
            ]
        )

        df = df.with_columns(
            [
                (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram"),
            ]
        )

        # Bollinger Bands
        bollinger_features = []
        for window in [10, 20]:
            bollinger_features.extend(
                [
                    ((pl.col("Close") - pl.col(f"sma_{window}")) / pl.col(f"volatility_{window}d")).alias(f"bollinger_position_{window}"),
                    (pl.col(f"volatility_{window}d") / pl.col(f"sma_{window}")).alias(f"bollinger_width_{window}"),
                ]
            )

        df = df.with_columns(bollinger_features)

        # ================================
        # VOLUME INDICATORS
        # ================================

        volume_features = []
        for window in [5, 10, 20]:
            volume_features.extend(
                [
                    # Volume ratios
                    (pl.col("Volume") / pl.col(f"volume_sma_{window}")).alias(f"volume_ratio_{window}"),
                    # Price-Volume correlation
                    pl.rolling_corr(a=pl.col("Close"), b=pl.col("Volume"), window_size=window).over("instr").alias(f"price_volume_corr_{window}"),
                    # On-Balance Volume
                    pl.when(pl.col("Close") > pl.col("close_lag_1"))
                    .then(pl.col("Volume"))
                    .when(pl.col("Close") < pl.col("close_lag_1"))
                    .then(-pl.col("Volume"))
                    .otherwise(0)
                    .rolling_sum(window_size=window)
                    .over("instr")
                    .alias(f"obv_{window}"),
                ]
            )

        df = df.with_columns(volume_features)

        # VWAP
        vwap_features = [
            ((pl.col("typical_price") * pl.col("Volume")).rolling_sum(20).over("instr") / pl.col("Volume").rolling_sum(20).over("instr")).alias(
                "vwap_20"
            ),
        ]
        df = df.with_columns(vwap_features)

        df = df.with_columns(
            [
                (pl.col("Close") / pl.col("vwap_20")).alias("close_vwap_ratio"),
            ]
        )

        # ================================
        # AUTOCORRELATION AND SERIAL DEPENDENCE
        # ================================

        autocorr_features = [
            # Return autocorrelations
            pl.rolling_corr(a=pl.col("return_1d"), b=pl.col("return_1d_lag_1"), window_size=20).over("instr").alias("return_autocorr_lag1"),
            pl.rolling_corr(a=pl.col("return_1d"), b=pl.col("return_1d_lag_5"), window_size=20).over("instr").alias("return_autocorr_lag5"),
            # Volume autocorrelations
            pl.rolling_corr(a=pl.col("Volume"), b=pl.col("volume_lag_1"), window_size=20).over("instr").alias("volume_autocorr_lag1"),
            # Cross-correlations
            pl.rolling_corr(a=pl.col("return_1d"), b=pl.col("Volume"), window_size=20).over("instr").alias("return_volume_corr"),
            pl.rolling_corr(a=abs(pl.col("return_1d")), b=pl.col("Volume"), window_size=20).over("instr").alias("abs_return_volume_corr"),
            # Pattern frequencies
            pl.when((pl.col("return_1d") > 0) & (pl.col("return_1d_lag_1") > 0))
            .then(1)
            .otherwise(0)
            .rolling_mean(20)
            .over("instr")
            .alias("positive_run_frequency"),
            pl.when((pl.col("return_1d") > 0) & (pl.col("return_1d_lag_1") < 0))
            .then(1)
            .otherwise(0)
            .rolling_mean(20)
            .over("instr")
            .alias("reversal_frequency"),
        ]

        df = df.with_columns(autocorr_features)

        # ================================
        # ADVANCED VOLATILITY MEASURES
        # ================================

        advanced_vol_features = [
            # Realized measures
            pl.col("return_1d").pow(2).rolling_mean(20).over("instr").alias("realized_variance_20d"),
            abs(pl.col("return_1d")).rolling_mean(20).over("instr").alias("realized_abs_deviation_20d"),
            # Volatility of volatility
            abs(pl.col("return_1d")).rolling_std(20).over("instr").alias("volatility_of_volatility"),
            # Range-based estimators
            ((pl.col("log_high") - pl.col("log_low")).pow(2) / (4 * np.log(2))).alias("rogers_satchell_volatility"),
            ((pl.col("log_high") - pl.col("log_low")).pow(2) / (4 * np.log(2))).rolling_mean(20).over("instr").alias("parkinson_volatility_20d"),
            # Garman-Klass
            ((pl.col("High") / pl.col("Low")) - 1).alias("garman_klass_volatility"),
        ]

        df = df.with_columns(advanced_vol_features)

        # ================================
        # PATTERN RECOGNITION
        # ================================

        pattern_features = [
            # Gap indicators
            ((pl.col("Open") - pl.col("close_lag_1")) / pl.col("close_lag_1")).alias("gap_pct"),
            # Candlestick patterns
            (abs(pl.col("Open") - pl.col("Close")) / pl.col("true_range_raw")).alias("doji_ratio"),
            ((pl.col("Close") - pl.col("Low")) / pl.col("true_range_raw")).alias("lower_wick_ratio"),
            ((pl.col("High") - pl.col("Close")) / pl.col("true_range_raw")).alias("upper_wick_ratio"),
            # Bar patterns
            pl.when((pl.col("High") < pl.col("High").shift(1).over("instr")) & (pl.col("Low") > pl.col("Low").shift(1).over("instr")))
            .then(1)
            .otherwise(0)
            .alias("inside_bar"),
            pl.when((pl.col("High") > pl.col("High").shift(1).over("instr")) & (pl.col("Low") < pl.col("Low").shift(1).over("instr")))
            .then(1)
            .otherwise(0)
            .alias("outside_bar"),
        ]

        df = df.with_columns(pattern_features)

        # ================================
        # DERIVED FEATURES FROM BASE COMPUTATIONS
        # ================================

        derived_features = [
            # Tail measures using pre-computed quantiles
            (pl.col("return_q05_60d") - pl.col("return_q95_60d")).alias("return_tail_spread_60d"),
            (pl.col("return_q01_60d") / pl.col("return_variance_60d").sqrt()).alias("left_tail_risk"),
            (pl.col("return_q99_60d") / pl.col("return_variance_60d").sqrt()).alias("right_tail_risk"),
            # Price position features
            (pl.col("Close") / pl.col("Close").rolling_max(60).over("instr")).alias("close_max_ratio_60d"),
            (pl.col("Close") / pl.col("close_min_60d")).alias("close_min_ratio_60d"),
            # Momentum features
            (pl.col("momentum_20d_rank") / 100).alias("momentum_score_20d"),
            (pl.col("momentum_60d_rank") / 100).alias("momentum_score_60d"),
            # Mean reversion features
            (pl.col("close_sma_20_ratio") - 1).alias("price_vs_ma20"),
            (pl.col("close_sma_50_ratio") - 1).alias("price_vs_ma50"),
            # Relative strength
            (pl.col("return_20d") - pl.col("return_20d").median().over(CONFIG.DATE_COL)).alias("relative_strength_20d"),
            # Liquidity measures
            (1 / pl.col("amihud_illiquidity")).alias("liquidity_measure"),
            (pl.col("dollar_volume") / pl.col("hl_range_pct")).alias("liquidity_ratio"),
            # Turnover measures
            pl.col("volume_ratio_20").alias("relative_turnover"),
            pl.col("volume_sma_5") / pl.col("volume_sma_20").alias("turnover_acceleration"),
            # Beta-like measure
            pl.rolling_corr(a=pl.col("return_1d"), b=pl.col("market_return"), window_size=60).over("instr").alias("market_beta_60d"),
            # Drawdown
            (pl.col("Close") / pl.col("Close").rolling_max(60).over("instr") - 1).alias("drawdown_60d"),
            # Regime indicators
            pl.when(pl.col("realized_variance_20d") > pl.col("realized_variance_20d").rolling_quantile(quantile=0.8, window_size=100).over("instr"))
            .then(2)
            .when(pl.col("realized_variance_20d") < pl.col("realized_variance_20d").rolling_quantile(quantile=0.2, window_size=100).over("instr"))
            .then(0)
            .otherwise(1)
            .alias("volatility_regime"),
            pl.when((pl.col("close_sma_20_ratio") - 1) > 0.05)
            .then(1)
            .when((pl.col("close_sma_20_ratio") - 1) < -0.05)
            .then(-1)
            .otherwise(0)
            .alias("trend_regime"),
            # Attention measures
            pl.when(pl.col("Volume") > pl.col("volume_sma_20") * 2).then(1).otherwise(0).alias("high_attention_day"),
            # Information flow proxies
            abs(pl.col("return_autocorr_lag1")).alias("price_inefficiency"),
            (pl.col("Volume") * abs(pl.col("return_1d"))).alias("informed_trading_proxy"),
            # Jump detection
            pl.when(abs(pl.col("return_1d")) > 3 * pl.col("return_std_60d")).then(1).otherwise(0).alias("jump_indicator"),
            # Extreme events
            pl.when(pl.col("return_1d") < pl.col("return_q01_60d")).then(1).otherwise(0).alias("extreme_negative_return"),
            pl.when(pl.col("return_1d") > pl.col("return_q99_60d")).then(1).otherwise(0).alias("extreme_positive_return"),
        ]

        df = df.with_columns(derived_features)

        return df

    def create_lag_features(self) -> pl.LazyFrame:
        lag_arr = self.df.drop(CONFIG.DATE_COL).to_numpy()

        return (
            pl.LazyFrame(
                np.vstack(
                    (
                        np.mean(lag_arr, axis=1),
                        np.median(lag_arr, axis=1),
                        np.max(lag_arr, axis=1),
                        np.min(lag_arr, axis=1),
                        np.percentile(lag_arr, 5, axis=1),
                        np.percentile(lag_arr, 95, axis=1),
                        np.std(lag_arr, axis=1),
                        np.where(lag_arr > 0, 1, 0).sum(axis=1) / lag_arr.shape[1],
                        np.where(lag_arr < 0, 1, 0).sum(axis=1) / lag_arr.shape[1],
                    )
                ).T,
                schema=["lags_mean", "lags_median", "lags_max", "lags_min", "lags_5_pct", "lags_95_pct", "lag_std", "lag_pos", "lag_neg"],
            )
            .with_columns(self.df.select(CONFIG.DATE_COL))
            .join(self.df.lazy(), how="left", on=CONFIG.DATE_COL)
        )

    def create_all_features(self) -> pl.LazyFrame:
        all = self.df.pipe(self.create_market_features)
        return (
            all.fill_null(strategy="mean")
            .fill_nan(0)
            .with_columns(
                *[
                    pl.when(pl.col(col).is_infinite()).then(0).otherwise(pl.col(col)).alias(col)
                    for col in all.collect_schema().names()
                    if all.collect_schema()[col] != pl.String
                ]
            )
            .drop(CONFIG.BASE_PRICE_FEATURES + CONFIG.BASE_OTHER_FEATURES)
        )

    def create_time_features(self, index_col: str = CONFIG.DATE_COL) -> pl.LazyFrame:
        """
        Create comprehensive temporal features from numerical index
        """
        return (
            self.df.pipe(self._basic_temporal_features, index_col)
            .pipe(self._cyclical_features)
            .pipe(self._market_calendar_features)
            .pipe(self._seasonal_features)
            .pipe(self._trading_session_features)
            .pipe(self._business_cycle_features)
            .pipe(self.cast_bool)
        )
