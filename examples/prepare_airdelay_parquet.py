"""Convert the large airdelay CSV into a compact DLSA Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.spark import get_spark
from dstats.spark import standardize_columns
from dstats.spark import write_single_parquet


DEFAULT_CSV = Path("/data/fli/carbon/running/data/airdelay_small.csv")
DEFAULT_PARQUET = Path("data/airdelay_small.parquet")
FEATURE_COLS = [
    "Year",
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "DepTime",
    "CRSDepTime",
    "CRSArrTime",
    "ActualElapsedTime",
    "Distance",
    "DepDelay",
]

AIRDELAY_SCHEMA = StructType(
    [
        StructField("Year", IntegerType()),
        StructField("Month", IntegerType()),
        StructField("DayofMonth", IntegerType()),
        StructField("DayOfWeek", IntegerType()),
        StructField("DepTime", DoubleType()),
        StructField("CRSDepTime", DoubleType()),
        StructField("ArrTime", DoubleType()),
        StructField("CRSArrTime", DoubleType()),
        StructField("UniqueCarrier", StringType()),
        StructField("FlightNum", StringType()),
        StructField("TailNum", StringType()),
        StructField("ActualElapsedTime", DoubleType()),
        StructField("CRSElapsedTime", DoubleType()),
        StructField("AirTime", DoubleType()),
        StructField("ArrDelay", DoubleType()),
        StructField("DepDelay", DoubleType()),
        StructField("Origin", StringType()),
        StructField("Dest", StringType()),
        StructField("Distance", DoubleType()),
        StructField("TaxiIn", DoubleType()),
        StructField("TaxiOut", DoubleType()),
        StructField("Cancelled", IntegerType()),
        StructField("CancellationCode", StringType()),
        StructField("Diverted", IntegerType()),
        StructField("CarrierDelay", DoubleType()),
        StructField("WeatherDelay", DoubleType()),
        StructField("NASDelay", DoubleType()),
        StructField("SecurityDelay", DoubleType()),
        StructField("LateAircraftDelay", DoubleType()),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--mode", choices=["error", "overwrite", "ignore"], default="overwrite")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(args.csv)
    spark = get_spark(
        "dstats-prepare-airdelay-parquet",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        raw = (
            spark.read.schema(AIRDELAY_SCHEMA)
            .option("header", "true")
            .option("nullValue", "NA")
            .option("nanValue", "NA")
            .csv(str(args.csv))
        )
        data = (
            raw.select(
                *[F.col(col).cast("double").alias(col) for col in FEATURE_COLS],
                F.col("ArrDelay").cast("double").alias("ArrDelay"),
            )
            .dropna(subset=[*FEATURE_COLS, "ArrDelay"])
        )

        data = standardize_columns(data, FEATURE_COLS, prefix="", keep_original=False)

        output = data.select("ArrDelay", *FEATURE_COLS)
        rows = write_single_parquet(output, args.out, mode=args.mode)
    finally:
        spark.stop()
    if rows is None:
        print(f"{args.out} already exists")
    else:
        print(f"Wrote {rows} rows to {args.out}")


if __name__ == "__main__":
    main()
