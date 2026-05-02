"""DARIMA example using the bundled electricity demand data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.darima import darima_forecast
from dstats.darima import darima_mapreduce
from dstats.darima import fit_darima_partitions
from dstats.darima import model_eval
from dstats.spark import get_spark


DEFAULT_DATA = Path("data/electricity.parquet")


def load_electricity(spark, path: Path, series: str, *, train_days: int, horizon: int, partition_num: int):
    if not path.exists():
        raise FileNotFoundError(path)
    train_rows = train_days * 24

    source = spark.read.parquet(str(path)).where(f"series = '{series}'")
    train = (
        source.where("split = 'train'")
        .orderBy("time")
        .select("time", "demand")
        .tail(train_rows)
    )
    test = (
        source.where("split = 'test'")
        .orderBy("time")
        .select("time", "demand")
        .limit(horizon)
        .toPandas()
    )
    train = spark.createDataFrame(train).toPandas()

    partition_size = int(np.ceil(len(train) / partition_num))
    train["partition_id"] = np.minimum(
        np.arange(len(train), dtype=np.int64) // partition_size,
        partition_num - 1,
    )
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--series", default="TOTAL")
    parser.add_argument("--train-days", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--partitions", type=int, default=4)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--tol", type=int, default=24)
    args = parser.parse_args()

    spark = get_spark(
        "dstats-darima-electricity",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        train, test = load_electricity(
            spark,
            args.path,
            args.series,
            train_days=args.train_days,
            horizon=args.horizon,
            partition_num=args.partitions,
        )
        sdf = spark.createDataFrame(train)
        mapped = fit_darima_partitions(
            sdf,
            value_col="demand",
            time_col="time",
            period=args.period,
            tol=args.tol,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            max_order=4,
            max_d=1,
            max_D=1,
            stepwise=True,
            approximation=False,
            nmodels=30,
        )
        assert mapped.count() == args.partitions

        reduced = darima_mapreduce(mapped, sample_size=len(train))
        assert reduced.shape == (args.tol + 2, args.tol + 3)

        forecast = darima_forecast(
            reduced["Theta_tilde"],
            reduced.iloc[:, 1:],
            train["demand"],
            period=args.period,
            h=args.horizon,
        )
        assert forecast.shape == (args.horizon, 3)

        scores = model_eval(
            train["demand"],
            test["demand"],
            period=args.period,
            pred=forecast["pred"],
            lower=forecast["lower"],
            upper=forecast["upper"],
        )
        assert scores.shape == (args.horizon, 3)
    finally:
        spark.stop()

    means = scores.mean().to_dict()
    print(
        "DARIMA electricity check passed: "
        f"series={args.series}, train_rows={len(train)}, horizon={args.horizon}, "
        f"mase={means['mase']:.4f}, smape={means['smape']:.4f}, msis={means['msis']:.4f}"
    )


if __name__ == "__main__":
    main()
