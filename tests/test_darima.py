from __future__ import annotations

import numpy as np

from dstats.darima import ar_coefficients
from dstats.darima import darima_forecast
from dstats.darima import darima_mapreduce
from dstats.darima import fit_darima_partitions
from dstats.darima import simulate_ar1


def test_ar_coefficients_for_ar1():
    coef = ar_coefficients(ar=[0.5], tol=4)

    assert coef.shape == (6,)
    assert coef[0] == 0.0
    assert coef[1] == 0.0
    assert coef[2:].tolist() == [0.5, 0.0, 0.0, 0.0]


def test_darima_synthetic_path(spark):
    sample_size = 240
    partition_num = 3
    tol = 5
    horizon = 3
    pdf = simulate_ar1(
        sample_size=sample_size,
        partition_num=partition_num,
        seed=13,
    )
    sdf = spark.createDataFrame(pdf)

    mapped = fit_darima_partitions(
        sdf,
        value_col="value",
        time_col="time",
        tol=tol,
        max_p=2,
        max_q=2,
        max_order=3,
        stepwise=True,
    )
    assert mapped.count() == partition_num

    reduced = darima_mapreduce(mapped, sample_size=sample_size)
    forecast = darima_forecast(
        reduced["Theta_tilde"],
        reduced.iloc[:, 1:],
        pdf["value"].iloc[:-horizon],
        h=horizon,
    )

    assert reduced.shape == (tol + 2, tol + 3)
    assert forecast.shape == (horizon, 3)
    assert np.isfinite(forecast.to_numpy()).all()
