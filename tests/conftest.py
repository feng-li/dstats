from __future__ import annotations

import os

import pytest

from dstats.spark import get_spark


@pytest.fixture(scope="session")
def spark():
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
    session = get_spark(
        "dstats-tests",
        master="local[2]",
        configs={
            "spark.sql.shuffle.partitions": "2",
            "spark.ui.enabled": "false",
        },
    )
    try:
        yield session
    finally:
        session.stop()
