"""Convert the cleaned used-car CSV into a compact raw Parquet dataset."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.spark import get_spark


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
    if args.coalesce <= 0:
        raise ValueError("coalesce must be positive")
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
        rows = output.count()
        output.coalesce(args.coalesce).write.mode("error").parquet(str(tmp_dir))

        part_files = sorted(tmp_dir.glob("part-*.parquet"))
        if args.coalesce == 1:
            if len(part_files) != 1:
                raise RuntimeError(f"Expected one Parquet part file, found {len(part_files)}")
            shutil.move(str(part_files[0]), args.out)
        else:
            shutil.move(str(tmp_dir), args.out)
            tmp_dir = args.out
    finally:
        spark.stop()
        if tmp_dir.exists() and tmp_dir != args.out:
            _remove_path(tmp_dir)

    print(f"Wrote {rows} rows to {args.out}")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


if __name__ == "__main__":
    main()
