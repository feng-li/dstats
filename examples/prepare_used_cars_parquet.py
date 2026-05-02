"""Convert the cleaned used-car CSV into a compact raw Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.spark import get_spark
from dstats.spark import write_single_parquet


DEFAULT_CSV = Path("/data/fli/carbon/running/data/used_cars_data/used_cars_data_clean.csv")
DEFAULT_PARQUET = Path("data/used_cars.parquet")
RAW_FEATURES = [
    "mileage",
    "year",
    "horsepower",
    "engine_displacement",
    "daysonmarket",
    "seller_rating",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--mode", choices=["error", "overwrite", "ignore"], default="overwrite")
    parser.add_argument("--coalesce", type=int, default=1)
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    spark = get_spark(
        "dstats-prepare-used-cars-parquet",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        raw = (
            spark.read.option("header", "true")
            .option("mode", "DROPMALFORMED")
            .csv(str(args.csv))
        )
        numeric_cols = [*RAW_FEATURES, "price"]
        data = (
            raw.select(*[F.col(col).cast("double").alias(col) for col in numeric_cols])
            .where(F.col("price") > 0)
            .dropna(subset=numeric_cols)
        )

        output = data.select("price", *RAW_FEATURES)
        rows = write_single_parquet(
            output,
            args.out,
            mode=args.mode,
            coalesce=args.coalesce,
        )
    finally:
        spark.stop()
    if rows is None:
        print(f"{args.out} already exists")
    else:
        print(f"Wrote {rows} rows to {args.out}")


if __name__ == "__main__":
    main()
