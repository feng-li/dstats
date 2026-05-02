"""Small Spark 4 smoke check for the Python-native DARIMA path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dstats.darima import darima_forecast
from dstats.darima import darima_mapreduce
from dstats.darima import fit_darima_partitions
from dstats.darima import model_eval
from dstats.darima import simulate_ar1
from dstats.spark import get_spark


def main() -> None:
    spark = get_spark(
        "dstats-darima-smoke",
        master="local[2]",
        configs={"spark.ui.enabled": "false"},
    )

    try:
        sample_size = 400
        partition_num = 4
        tol = 8
        horizon = 5

        pdf = simulate_ar1(
            sample_size=sample_size,
            partition_num=partition_num,
            seed=11,
        )
        sdf = spark.createDataFrame(pdf)

        mapped = fit_darima_partitions(
            sdf,
            value_col="value",
            time_col="time",
            tol=tol,
            max_p=3,
            max_q=3,
            max_order=4,
            stepwise=True,
        )
        assert mapped.count() == partition_num

        reduced = darima_mapreduce(mapped, sample_size=sample_size)
        assert reduced.shape == (tol + 2, tol + 3)

        train = pdf["value"].iloc[:-horizon]
        test = pdf["value"].iloc[-horizon:]
        forecast = darima_forecast(
            reduced["Theta_tilde"],
            reduced.iloc[:, 1:],
            train,
            h=horizon,
        )
        assert forecast.shape == (horizon, 3)

        scores = model_eval(
            train,
            test,
            period=1,
            pred=forecast["pred"],
            lower=forecast["lower"],
            upper=forecast["upper"],
        )
        assert scores.shape == (horizon, 3)
    finally:
        spark.stop()

    print("DARIMA smoke check passed")


if __name__ == "__main__":
    main()
