from dataclasses import dataclass
import polars as pl
from CONFIG import CONFIG


@dataclass
class PREPROCESSOR:
    """
    Preprocessor for Polars LazyFrame containing financial time-series data.

    Provides methods for inspecting schema, filling missing values, and cleaning the data.

    Parameters
    ----------
    df : pl.LazyFrame
        The input Polars LazyFrame to preprocess. Should contain a date column
        specified by CONFIG.DATE_COL and feature columns such as prices, volume,
        and open interest.

    Attributes
    ----------
    df : pl.LazyFrame
        The dataframe being preprocessed.
    """

    df: pl.LazyFrame

    def get_schema(self) -> list[str]:
        """
        Get the list of column names in the dataframe.

        Returns
        -------
        list of str
            Column names in the dataframe.
        """
        return self.df.collect_schema().names()

    def fill_null(self) -> None:
        """
        Fill missing values in the dataframe with appropriate strategies.

        Notes
        -----
        - Price and open interest columns are forward-filled assuming values remain
          unchanged if missing.
        - Volume columns ('Volume' and 'adj_volume') are filled with 0.0, assuming
          no trading occurred.
        """
        # Identify volume columns
        volume_cols = [col for col in self.get_schema() if col in ["Volume", "adj_volume"]]

        self.df = self.df.with_columns(
            # Forward fill all columns except date and volume columns
            pl.all().exclude(columns=[CONFIG.DATE_COL] + volume_cols).fill_null(strategy="forward"),
            # Fill volume columns with 0.0
            pl.col(volume_cols).fill_null(0.0),
        )

    def clean(self) -> pl.LazyFrame:
        """
        Clean the dataframe by handling missing values and converting column types.

        Steps
        -----
        1. Call fill_null() to handle missing values.
        2. Fill any remaining nulls with 0.
        3. Cast all columns except the date column to Float64.

        Returns
        -------
        pl.LazyFrame
            The cleaned LazyFrame ready for feature engineering.
        """
        self.fill_null()
        self.df = self.df.fill_null(0).with_columns(pl.all().exclude(CONFIG.DATE_COL).cast(pl.Float64))
        return self.df 