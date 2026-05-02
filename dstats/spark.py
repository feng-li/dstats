"""Small Spark helpers for migrated code."""

from __future__ import annotations

from collections.abc import Mapping

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
