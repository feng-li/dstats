"""Small Spark helpers for migrated code."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from math import isfinite
from pathlib import Path
import shutil

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import SparkSession


def get_spark(
    app_name: str = "dstats",
    master: str | None = None,
    configs: Mapping[str, str] | None = None,
    enable_arrow: bool = True,
) -> SparkSession:
    """Create or reuse a Spark session with current PySpark defaults."""

    builder = SparkSession.builder.appName(app_name)

    if master:
        builder = builder.master(master)

    if enable_arrow:
        builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
        builder = builder.config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

    for key, value in (configs or {}).items():
        builder = builder.config(key, value)

    return builder.getOrCreate()


def standardize_columns(
    sdf: DataFrame,
    columns: Sequence[str],
    *,
    prefix: str = "z_",
    keep_original: bool = True,
) -> DataFrame:
    """Add or replace standardized Spark DataFrame columns."""

    if not columns:
        return sdf

    stats_row = sdf.agg(
        *[F.mean(col).alias(f"{col}__mean") for col in columns],
        *[F.stddev_samp(col).alias(f"{col}__std") for col in columns],
    ).collect()[0]

    out = sdf
    for col in columns:
        mean_value = stats_row[f"{col}__mean"]
        std_value = stats_row[f"{col}__std"]
        if mean_value is None or std_value is None:
            raise ValueError(f"Column {col!r} has no complete values to standardize")
        mean = float(mean_value)
        std = float(std_value)
        if not isfinite(std) or std <= 0:
            raise ValueError(f"Column {col!r} has non-positive standard deviation")

        target_col = f"{prefix}{col}"
        out = out.withColumn(target_col, ((F.col(col) - F.lit(mean)) / F.lit(std)).cast("double"))
        if not keep_original and target_col != col:
            out = out.drop(col)
    return out


def with_partition_id(
    sdf: DataFrame,
    partition_num: int,
    *,
    column: str = "partition_id",
) -> DataFrame:
    """Add a deterministic-enough local partition id column for examples."""

    if partition_num <= 0:
        raise ValueError("partition_num must be positive")
    return sdf.withColumn(
        column,
        F.pmod(F.monotonically_increasing_id(), F.lit(partition_num)).cast("long"),
    )


def write_single_parquet(
    sdf: DataFrame,
    path: str | Path,
    *,
    mode: str = "overwrite",
    coalesce: int = 1,
) -> int | None:
    """Write Spark output as one Parquet file by default.

    Returns the row count written, or ``None`` when ``mode='ignore'`` and the
    output already exists. When ``coalesce`` is greater than one, the output is a
    normal Parquet directory.
    """

    if mode not in {"error", "overwrite", "ignore"}:
        raise ValueError("mode must be one of: error, overwrite, ignore")
    if coalesce <= 0:
        raise ValueError("coalesce must be positive")

    out_path = Path(path)
    if out_path.exists():
        if mode == "error":
            raise FileExistsError(out_path)
        if mode == "ignore":
            return None
        remove_path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_path.parent / f".{out_path.name}.spark-tmp"
    if tmp_dir.exists():
        remove_path(tmp_dir)

    rows = sdf.count()
    try:
        sdf.coalesce(coalesce).write.mode("error").parquet(str(tmp_dir))

        if coalesce == 1:
            part_files = sorted(tmp_dir.glob("part-*.parquet"))
            if len(part_files) != 1:
                raise RuntimeError(f"Expected one Parquet part file, found {len(part_files)}")
            shutil.move(str(part_files[0]), out_path)
        else:
            shutil.move(str(tmp_dir), out_path)
            tmp_dir = out_path
    finally:
        if tmp_dir.exists() and tmp_dir != out_path:
            remove_path(tmp_dir)
    return rows


def remove_path(path: str | Path) -> None:
    """Remove a file or directory path."""

    target = Path(path)
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


__all__ = [
    "get_spark",
    "remove_path",
    "standardize_columns",
    "with_partition_id",
    "write_single_parquet",
]
