"""DQR example using the raw used-car Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.dqr import dqr_fit
from dstats.dqr import fit_quantile_pilot
from dstats.spark import get_spark
from dstats.spark import standardize_columns
from dstats.spark import with_partition_id


DEFAULT_DATA = Path("data/used_cars.parquet")
RAW_FEATURES = [
    "mileage",
    "year",
    "horsepower",
    "engine_displacement",
    "daysonmarket",
    "seller_rating",
]
FEATURE_COLS = ["intercept", *[f"z_{col}" for col in RAW_FEATURES]]
LABEL_COL = "log_price"


def load_used_cars(
    spark,
    path: Path,
    *,
    nrows: int,
    partitions: int,
) -> tuple[DataFrame, list[str], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    if nrows <= 0:
        raise ValueError("nrows must be positive")
    if partitions <= 0:
        raise ValueError("partitions must be positive")

    selected = _read_raw_data(spark, path, nrows=nrows).cache()
    sample_size = selected.count()
    if sample_size <= len(FEATURE_COLS):
        raise ValueError(f"Only {sample_size} complete used-car rows were available")

    prepared = standardize_columns(selected, RAW_FEATURES, prefix="z_")
    prepared = prepared.withColumn(LABEL_COL, F.log("price")).withColumn("intercept", F.lit(1.0))
    prepared = with_partition_id(prepared.select(LABEL_COL, *FEATURE_COLS), partitions)
    return prepared, FEATURE_COLS, sample_size


def _read_raw_data(spark, path: Path, *, nrows: int) -> DataFrame:
    numeric_cols = [*RAW_FEATURES, "price"]
    if path.suffix == ".parquet":
        return (
            spark.read.parquet(str(path))
            .select(*numeric_cols)
            .where(F.col("price") > 0)
            .dropna(subset=numeric_cols)
            .limit(nrows)
        )

    raw = (
        spark.read.option("header", "true")
        .option("mode", "DROPMALFORMED")
        .csv(str(path))
    )
    return (
        raw.select(*[F.col(col).cast("double").alias(col) for col in numeric_cols])
        .where(F.col("price") > 0)
        .dropna(subset=numeric_cols)
        .limit(nrows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--nrows", type=int, default=5_000)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--quantile", type=float, default=0.5)
    parser.add_argument("--pilot-fraction", type=float, default=0.35)
    args = parser.parse_args()

    spark = get_spark(
        "dstats-dqr-used-cars",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        sdf, feature_cols, sample_size = load_used_cars(
            spark,
            args.path,
            nrows=args.nrows,
            partitions=args.partitions,
        )
        fitted = dqr_fit(
            sdf,
            label_col=LABEL_COL,
            feature_cols=feature_cols,
            quantile=args.quantile,
            pilot_fraction=args.pilot_fraction,
            pilot_seed=19,
        )
        assert fitted.shape == (len(feature_cols), 5)
        assert np.isfinite(fitted["beta_dqr"]).all()
        assert np.isfinite(fitted["se_dqr"]).all()

        full_pdf = sdf.select(*feature_cols, LABEL_COL).toPandas()
        full_fit = fit_quantile_pilot(
            full_pdf,
            label_col=LABEL_COL,
            feature_cols=feature_cols,
            quantile=args.quantile,
        )
        full_beta = np.asarray(full_fit.params, dtype=float)
        max_abs_diff = float(np.max(np.abs(fitted["beta_dqr"].to_numpy() - full_beta)))
    finally:
        spark.stop()

    print(
        "DQR used-car check passed: "
        f"rows={sample_size}, quantile={args.quantile}, "
        f"terms={len(feature_cols)}, max_abs_diff_vs_full={max_abs_diff:.4f}"
    )
    print(fitted[["beta_dqr", "beta_pilot", "se_dqr"]].round(4).to_string())


if __name__ == "__main__":
    main()
