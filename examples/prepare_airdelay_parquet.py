"""Convert the large airdelay CSV into a DLSA-ready Parquet dataset."""

from __future__ import annotations

import argparse
import shutil
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
    if args.out.exists():
        if args.mode == "error":
            raise FileExistsError(args.out)
        if args.mode == "ignore":
            print(f"{args.out} already exists")
            return
        _remove_path(args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out.parent / f".{args.out.name}.spark-tmp"
    if tmp_dir.exists():
        _remove_path(tmp_dir)

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
                (F.col("ArrDelay") > 0).cast("long").alias("nominal_delay"),
                (F.col("ArrDelay") > 20).cast("long").alias("real_delay"),
            )
            .dropna(subset=[*FEATURE_COLS, "nominal_delay", "real_delay"])
        )

        stats_exprs = []
        for col in FEATURE_COLS:
            stats_exprs.append(F.mean(col).alias(f"{col}__mean"))
            stats_exprs.append(F.stddev_samp(col).alias(f"{col}__std"))
        stats = data.agg(*stats_exprs).collect()[0].asDict()

        for col in FEATURE_COLS:
            mean = stats[f"{col}__mean"]
            std = stats[f"{col}__std"]
            if std is None or std == 0:
                raise ValueError(f"Feature {col!r} has zero variance")
            data = data.withColumn(col, ((F.col(col) - F.lit(mean)) / F.lit(std)).cast("double"))

        output = data.select("nominal_delay", "real_delay", *FEATURE_COLS)
        rows = output.count()
        output.coalesce(1).write.mode("error").parquet(str(tmp_dir))
        part_files = sorted(tmp_dir.glob("part-*.parquet"))
        if len(part_files) != 1:
            raise RuntimeError(f"Expected one Parquet part file, found {len(part_files)}")
        shutil.move(str(part_files[0]), args.out)
    finally:
        spark.stop()
        if tmp_dir.exists():
            _remove_path(tmp_dir)

    print(f"Wrote {rows} rows to {args.out}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    main()
