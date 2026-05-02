"""Small DLSA check using the local compact airdelay Parquet dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.dlsa import dlsa_fit
from dstats.dlsa import dlsa_mapreduce
from dstats.dlsa import fit_logistic_partitions
from dstats.spark import get_spark
from dstats.spark import with_partition_id


DEFAULT_DATA = Path("data/airdelay_small.parquet")
DELAY_THRESHOLDS = {"nominal_delay": 0.0, "real_delay": 20.0}
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


def load_airdelay_sdf(
    spark,
    path: Path,
    *,
    nrows: int,
    partition_num: int,
    label_col: str,
) -> DataFrame:
    source = spark.read.parquet(str(path))
    missing = sorted(set(FEATURE_COLS).difference(source.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required feature columns: {missing}."
        )

    if label_col not in source.columns:
        if "ArrDelay" not in source.columns:
            raise ValueError(
                f"{path} must contain {label_col!r} or raw 'ArrDelay' so the delay label "
                "can be derived."
            )
        threshold = DELAY_THRESHOLDS[label_col]
        source = source.withColumn(label_col, (F.col("ArrDelay") > F.lit(threshold)).cast("long"))

    sdf = source.select(label_col, *FEATURE_COLS)
    if nrows > 0:
        sdf = sdf.limit(nrows)

    return with_partition_id(sdf, partition_num)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_DATA,
        help="Parquet file containing ArrDelay or a delay label plus the 10 feature columns.",
    )
    parser.add_argument("--nrows", type=int, default=5000)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument(
        "--label-col",
        choices=sorted(DELAY_THRESHOLDS),
        default="nominal_delay",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise FileNotFoundError(
            f"{args.path} does not exist. Run examples/prepare_airdelay_parquet.py first."
        )

    spark = get_spark(
        "dstats-dlsa-airdelay-small",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        sdf = load_airdelay_sdf(
            spark,
            args.path,
            nrows=args.nrows,
            partition_num=args.partitions,
            label_col=args.label_col,
        ).cache()
        sample_size = sdf.count()
        if sdf.select(args.label_col).distinct().count() != 2:
            raise ValueError("The sampled data must contain both delayed and non-delayed rows")

        mapped = fit_logistic_partitions(
            sdf,
            label_col=args.label_col,
            feature_cols=FEATURE_COLS,
        )
        mapped_count = mapped.count()
        assert mapped_count == len(FEATURE_COLS) * args.partitions

        reduced = dlsa_mapreduce(mapped)
        assert reduced.shape == (len(FEATURE_COLS), len(FEATURE_COLS) + 2)

        selected = dlsa_fit(
            reduced.iloc[:, 2:],
            reduced["beta_byOLS"],
            sample_size=sample_size,
        )
        assert selected.shape == (len(FEATURE_COLS), 2)
    finally:
        spark.stop()

    print(
        "DLSA airdelay check passed: "
        f"rows={sample_size}, features={len(FEATURE_COLS)}, label={args.label_col}"
    )


if __name__ == "__main__":
    main()
