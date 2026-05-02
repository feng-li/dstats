"""Merge bundled electricity series into one Parquet table."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.spark import get_spark


DEFAULT_INPUT = Path("darima/data")
DEFAULT_OUTPUT = Path("data/electricity.parquet")
ELECTRICITY_SCHEMA = StructType(
    [
        StructField("demand", DoubleType()),
        StructField("time", StringType()),
    ]
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["error", "overwrite", "ignore"], default="overwrite")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
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
        "dstats-prepare-electricity-parquet",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        frames = []
        for path in sorted(args.input.glob("*.csv")):
            series, split = _parse_electricity_name(path)
            frame = (
                spark.read.schema(ELECTRICITY_SCHEMA)
                .option("header", "true")
                .csv(str(path))
                .select(
                    F.lit(series).alias("series"),
                    F.lit(split).alias("split"),
                    F.to_timestamp("time").alias("time"),
                    F.col("demand").cast("double").alias("demand"),
                )
                .dropna(subset=["time", "demand"])
            )
            frames.append(frame)

        if not frames:
            raise ValueError(f"No CSV files found in {args.input}")

        output = frames[0]
        for frame in frames[1:]:
            output = output.unionByName(frame)
        output = output.select("series", "split", "time", "demand")
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


def _parse_electricity_name(path: Path) -> tuple[str, str]:
    stem = path.stem
    if stem.endswith("_train"):
        return stem[: -len("_train")], "train"
    if stem.endswith("_test"):
        return stem[: -len("_test")], "test"
    raise ValueError(f"Unexpected electricity filename: {path.name}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    main()
