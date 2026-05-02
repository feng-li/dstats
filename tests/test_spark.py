from __future__ import annotations

import pytest

from dstats.spark import standardize_columns
from dstats.spark import with_partition_id
from dstats.spark import write_single_parquet


def test_standardize_columns_adds_z_columns(spark):
    sdf = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])

    out = standardize_columns(sdf, ["x"]).orderBy("x")

    rows = out.collect()
    assert [row.x for row in rows] == [1.0, 2.0, 3.0]
    assert [pytest.approx(row.z_x) for row in rows] == [-1.0, 0.0, 1.0]


def test_standardize_columns_can_replace_columns(spark):
    sdf = spark.createDataFrame([(1.0,), (2.0,), (3.0,)], ["x"])

    out = standardize_columns(sdf, ["x"], prefix="", keep_original=False).orderBy("x")

    assert out.columns == ["x"]
    assert [pytest.approx(row.x) for row in out.collect()] == [-1.0, 0.0, 1.0]


def test_standardize_columns_rejects_constant_column(spark):
    sdf = spark.createDataFrame([(1.0,), (1.0,), (1.0,)], ["x"])

    with pytest.raises(ValueError, match="non-positive standard deviation"):
        standardize_columns(sdf, ["x"])


def test_with_partition_id_adds_bounded_ids(spark):
    sdf = spark.range(8)

    out = with_partition_id(sdf, 3)
    partition_ids = {row.partition_id for row in out.select("partition_id").collect()}

    assert partition_ids <= {0, 1, 2}
    assert out.count() == 8


def test_write_single_parquet_writes_file_and_supports_ignore(tmp_path, spark):
    sdf = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "label"])
    out = tmp_path / "sample.parquet"

    rows = write_single_parquet(sdf, out)

    assert rows == 2
    assert out.is_file()
    assert spark.read.parquet(str(out)).count() == 2
    assert write_single_parquet(sdf, out, mode="ignore") is None
