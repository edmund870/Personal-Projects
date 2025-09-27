from dataclasses import dataclass
import polars as pl
from CONFIG import CONFIG


@dataclass
class PREPROCESSOR:
    df: pl.LazyFrame

    def get_schema(self):
        return self.df.collect_schema()

    def fill_null(self):
        self.df = self.df.with_columns(
            # forward fill prices and open interest assuming unchanged
            pl.all()
            .exclude(
                columns=[CONFIG.DATE_COL]
                + [col for col in self.schema.keys() if col in ["Volume", "adj_volume"]]
            )
            .fill_null(strategy="forward"),
            # fill null volumes as 0.0 no trading
            pl.col(
                [col for col in self.schema.keys() if col in ["Volume", "adj_volume"]]
            ).fill_null(0.0),
        )

    def compute_returns(self):
        self.df = self.df.with_columns(
            [
                (pl.col(col) + 1).log().diff()
                for col in self.schema.keys()
                if any([i for i in CONFIG.RETURNS_FEATURES if i in col])
            ]
        )

    def process_LME(self):
        return (
            self.df.select(
                [CONFIG.DATE_COL] + [col for col in self.schema.keys() if "LME_" in col]
            )
            .unpivot(index=CONFIG.DATE_COL)
            .rename({"value": "Close"})
            .with_columns(
                [pl.col("Close").alias(col) for col in CONFIG.BASE_PRICE_FEATURES]
                + [pl.lit(0.0).alias(col) for col in CONFIG.BASE_OTHER_FEATURES]
                + [pl.lit("LME").alias("type")]
            )
            .with_columns(pl.col("variable").str.split("_").list.get(1).alias("instr"))
            .drop("variable")
            .collect()
        )

    def process_JPX(self):
        return (
            self.df.select(
                [CONFIG.DATE_COL] + [col for col in self.schema.keys() if "JPX_" in col]
            )
            .unpivot(index=CONFIG.DATE_COL)
            .with_columns(pl.col("variable").str.split("_").alias("temp"))
            .with_columns(
                pl.when(
                    pl.col("temp")
                    .list.get(-1)
                    .is_in(CONFIG.BASE_PRICE_FEATURES + ["Volume"])
                )
                .then(pl.col("temp").list.get(-1))
                .when(pl.col("temp").list.get(-1) == "interest")
                .then(pl.lit("open_interest"))
                .otherwise(pl.lit("settlement_price"))
                .alias("feature"),
                pl.col("variable")
                .map_elements(
                    lambda v: next((k for k in CONFIG.ASSETS["JPX"] if k in v), None),
                    return_dtype=pl.String,
                )
                .alias("instr"),
                pl.lit("JPX").alias("type"),
            )
            .drop(["temp", "variable"])
            .collect()
            .pivot(index=[CONFIG.DATE_COL, "instr", "type"], on="feature")
        )

    def process_FX(self):
        return (
            self.df.select(
                [CONFIG.DATE_COL] + [col for col in self.schema.keys() if "FX_" in col]
            )
            .unpivot(index=CONFIG.DATE_COL)
            .rename({"value": "Close"})
            .with_columns(
                [pl.col("Close").alias(col) for col in CONFIG.BASE_PRICE_FEATURES]
                + [pl.lit(0.0).alias(col) for col in CONFIG.BASE_OTHER_FEATURES]
                + [pl.lit("FX").alias("type")]
            )
            .with_columns(pl.col("variable").str.split("_").list.get(-1).alias("instr"))
            .drop("variable")
            .collect()
        )

    def process_stock(self):
        return (
            self.df.select(
                [CONFIG.DATE_COL]
                + [col for col in self.schema.keys() if "US_Stock" in col]
            )
            .unpivot(index=CONFIG.DATE_COL)
            .with_columns(pl.col("variable").str.split("_").alias("temp"))
            .with_columns(
                pl.when(
                    pl.col("temp")
                    .list.get(-1)
                    .str.to_titlecase()
                    .is_in(CONFIG.BASE_PRICE_FEATURES + ["Volume"])
                )
                .then(pl.col("temp").list.get(-1).str.to_titlecase())
                .when(pl.col("temp").list.get(-1) == "interest")
                .then(pl.lit("open_interest"))
                .otherwise(pl.lit("settlement_price"))
                .alias("feature"),
                (pl.col("variable").str.split("_").list.get(2).alias("instr")),
                pl.lit("US").alias("type"),
            )
            .drop(["temp", "variable"])
            .collect()
            .pivot(index=[CONFIG.DATE_COL, "instr", "type"], on="feature")
            .with_columns(
                pl.lit(0.0).alias("open_interest"),
            )
        )

    def clean(self):
        self.df = self.df.with_columns(
            pl.all().exclude(CONFIG.DATE_COL).cast(pl.Float64)
        )
        self.schema = self.get_schema()
        self.df = self.df.drop(
            [col for col in self.schema.keys() if "settlement_price" in col]
        )
        self.schema = self.get_schema()
        self.fill_null()

        return self.df

    def transform(self):
        # self.compute_returns()
        self.df = self.df.drop_nulls()
        return pl.concat(
            [
                self.process_LME(),
                self.process_JPX(),
                self.process_FX(),
                self.process_stock(),
            ],
            how="diagonal",
        ).sort(by=CONFIG.DATE_COL)
